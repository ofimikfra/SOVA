import cv2
import mediapipe as mp
import threading
import time
import asyncio
import json
import websockets

from src.screen_capture import getScreenFrame
from src.webcam_capture import getCameraFrame
from models.expression import detectExpression, face_mesh
from models.gesture_v4 import detectGesture
from models.body_action import detectBodyAction
from src.processor import processExpression, processGesture, processBodyAction, flushAll
from src.tts_engine import speak

from src import config as _config
import src.processor as _processor
import src.tts_engine as _tts
from src.ollama_manager import ensure_ollama

# ─────────────────────────────────────────────
#  Settings
# ─────────────────────────────────────────────

_current_config = _config.load()

def _apply_settings(partial: dict):
    global _current_config
    _current_config = _config.update(partial)

    if "flush_interval" in partial:
        _processor.set_interval(float(partial["flush_interval"]))

    if "tts_enabled" in partial:
        _tts.set_enabled(partial["tts_enabled"])

    if "ollama_model" in partial:
        # Pull in background so the UI doesn't block
        new_model = partial["ollama_model"]
        threading.Thread(
            target=ensure_ollama,
            args=(new_model,),
            daemon=True
        ).start()

    print(f"[CONFIG] Settings updated: {partial}")
    _broadcast_sync({"type": "config", **_current_config})



# ─────────────────────────────────────────────
#  WebSocket Server
# ─────────────────────────────────────────────

WS_HOST = "localhost"
WS_PORT = 8765

_ws_clients: set = set()
_caption_queue: list = []
_caption_lock = threading.Lock()
_ws_loop: asyncio.AbstractEventLoop | None = None


async def _ws_handler(websocket):
    _ws_clients.add(websocket)
    print(f"[WS] Extension connected — {len(_ws_clients)} client(s)")
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)

                if msg.get("type") == "caption":
                    text = msg.get("text", "").strip()
                    if text:
                        with _caption_lock:
                            _caption_queue.append(text)

                elif msg.get("type") == "settings":
                    # Strip the type key and apply the rest as settings
                    partial = {k: v for k, v in msg.items() if k != "type"}
                    _apply_settings(partial)

                elif msg.get("type") == "get_config":
                    # Dashboard just connected — send current config back
                    reply = json.dumps({"type": "config", **_current_config})
                    await websocket.send(reply)

            except json.JSONDecodeError:
                pass
    except (websockets.exceptions.ConnectionClosedOK,
            websockets.exceptions.ConnectionClosedError):
        pass
    finally:
        _ws_clients.discard(websocket)
        print(f"[WS] Extension disconnected — {len(_ws_clients)} client(s)")


async def _ws_broadcast(payload: dict):
    if not _ws_clients:
        return
    message = json.dumps(payload)
    dead = set()
    for ws in _ws_clients.copy():
        try:
            await ws.send(message)
        except websockets.exceptions.ConnectionClosed:
            dead.add(ws)
    _ws_clients.difference_update(dead)


def _broadcast_sync(payload: dict):
    if _ws_loop and _ws_clients:
        asyncio.run_coroutine_threadsafe(_ws_broadcast(payload), _ws_loop)


async def _ws_serve():
    async with websockets.serve(_ws_handler, WS_HOST, WS_PORT):
        print(f"[WS] Server listening on ws://{WS_HOST}:{WS_PORT}")
        await asyncio.get_event_loop().create_future()


def _start_ws_thread():
    global _ws_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _ws_loop = loop
    loop.run_until_complete(_ws_serve())


# ─────────────────────────────────────────────
#  Main Detection Loop
# ─────────────────────────────────────────────

def run_system(callback=None, source="webcam"):

    cfg   = _config.load()
    model = cfg.get("ollama_model", "llama3.2:3b")
    ollama_ready = ensure_ollama(model)
    if not ollama_ready:
        print("[SOVA] Continuing without Ollama — template descriptions will be used.")

    ws_thread = threading.Thread(target=_start_ws_thread, daemon=True)
    ws_thread.start()
    time.sleep(0.5)

    if source == "screen":
        get_frame = getScreenFrame
        mirror    = False
        print("[SOVA] Monitoring Screen...")
    else:
        get_frame = getCameraFrame
        mirror    = True
        print("[SOVA] Monitoring Webcam...")

    # ── initialise display state so overlay never crashes before first flush ──
    display_expr      = "Neutral"
    display_gest      = "No Gesture"
    display_act       = "Person Center"
    display_sentiment = "neutral"
    display_conf      = 0.0
    display_desc      = "Waiting for first analysis..."

    print("[SOVA] Engine Active. Press 'q' on the video window to stop.")

    while True:
        frame = get_frame()
        if frame is None:
            continue

        if mirror:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 1. Facial Expressions
        results = face_mesh.detect(image)
        raw_expr, expr_conf = "Neutral", 1.0
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                raw_expr, expr_conf = detectExpression(face_landmarks, h, w)
                xs = [lm.x * w for lm in face_landmarks]
                ys = [lm.y * h for lm in face_landmarks]
                cv2.rectangle(frame,
                    (int(min(xs)), int(min(ys))),
                    (int(max(xs)), int(max(ys))),
                    (0, 255, 0), 2)

        # 2. Gestures & Body Actions
        raw_gest,   gest_conf   = detectGesture(frame)
        raw_action, action_conf = detectBodyAction(frame)

        # 3. Feed Processor
        processExpression(raw_expr,   expr_conf)
        processGesture(raw_gest,      gest_conf)
        processBodyAction(raw_action, action_conf)

        # 4. Drain caption queue
        with _caption_lock:
            pending_captions = _caption_queue.copy()
            _caption_queue.clear()

        # 5. Flush every N seconds
        stable_results = flushAll(captions=pending_captions)
        if stable_results:
            expr, gest, act, sentiment, sent_conf, description = stable_results

            # Update display state
            display_expr      = expr
            display_gest      = gest
            display_act       = act
            display_sentiment = sentiment
            display_conf      = sent_conf
            display_desc      = description

            _broadcast_sync({
                "type":             "result",
                "expression":       expr,
                "gesture":          gest,
                "action":           act,
                "sentiment":        sentiment,
                "sentimentConf":    sent_conf,
                "summary":          description,      # dashboard reads msg.summary
            })

            speak(description)

        # 6. Visual Overlay — always uses display_ vars, never crashes
        overlay_lines = [
            f"Expression:  {display_expr}",
            f"Gesture:     {display_gest}",
            f"Action:      {display_act}",
            f"Sentiment:   {display_sentiment} ({display_conf:.0%})",
            f"Description: {display_desc}",
            f"WS clients:  {len(_ws_clients)}",
        ]
        for i, line in enumerate(overlay_lines):
            cv2.putText(frame, line, (20, h - 30 - (i * 35)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("SOVA", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_system(source="webcam")