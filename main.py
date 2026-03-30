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
import src.audio_capture        as _audio
import src.stt_engine as _stt

from src import config as _config
import src.processor as _processor
import src.tts_engine as _tts
from src.translator import translate, is_rtl

_current_config = _config.load()


def _apply_settings(partial: dict):
    global _current_config
    _current_config = _config.update(partial)

    if "flush_interval" in partial:
        _processor.set_interval(float(partial["flush_interval"]))

    if "tts_enabled" in partial:
        _tts.set_enabled(partial["tts_enabled"])

    print(f"[CONFIG] Settings updated: {partial}")
    _broadcast_sync({"type": "config", **_current_config})


# ─────────────────────────────────────────────
#  WebSocket Server
# ─────────────────────────────────────────────

WS_HOST = "localhost"
WS_PORT = 8765

_ws_clients: set = set()
_ws_loop: asyncio.AbstractEventLoop | None = None


async def _ws_handler(websocket):
    _ws_clients.add(websocket)
    print(f"[WS] Extension connected — {len(_ws_clients)} client(s)")
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
                if msg.get("type") == "settings":
                    _apply_settings(msg)
                elif msg.get("type") == "get_config":
                    await websocket.send(json.dumps(
                        {"type": "config", **_current_config}
                    ))
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

def run_system(callback=None, source="screen", headless=False,
               stop_event=None, ws_broadcast=None):
    # ws_broadcast: callable provided by app.py when running as desktop app.
    # When None we start our own WS server (standalone / CLI mode).

    # ── Load config fresh at engine start ─────
    # This ensures any settings saved while the engine was stopped
    # (via pywebview api or WS) are picked up before the loop runs.
    cfg = _config.load()
    _processor.set_interval(float(cfg.get("flush_interval", 30)))
    _tts.set_enabled(cfg.get("tts_enabled", True))
    _tts.set_volume(cfg.get("tts_volume", 0.25))
    print(f"[SOVA] Loaded config: interval={cfg.get('flush_interval')}s  "
          f"tts={'on' if cfg.get('tts_enabled') else 'off'}  "
          f"model={cfg.get('ollama_model')}")

    # ── WebSocket ─────────────────────────────────────────
    if ws_broadcast is None:
        ws_thread = threading.Thread(target=_start_ws_thread, daemon=True)
        ws_thread.start()
        time.sleep(0.5)
        broadcast = _broadcast_sync
    else:
        broadcast = ws_broadcast

    # ── Audio capture + transcription ─────────
    audio_ok = _audio.start()
    if audio_ok:
        _stt.start(_audio.get_queue())
    else:
        print("[SOVA] ⚠️  Audio capture unavailable — sentiment will be expression-only.")

    # ── Video source ──────────────────────────
    if source == "screen":
        get_frame = getScreenFrame
        mirror    = False
        print("[SOVA] Monitoring Screen...")
    else:
        get_frame = getCameraFrame
        mirror    = True
        print("[SOVA] Monitoring Webcam...")

    display_expr = "Neutral"
    display_gest = "No Gesture"
    display_act  = "Person Center"
    sentiment    = "neutral"
    sent_conf    = 1.0
    description  = ""

    accumulated_transcripts: list[str] = []

    print("[SOVA] Engine Active. Press 'q' on the video window to stop.")

    # Broadcast NOW — models loaded, loop starting, truly ready
    broadcast({"type": "engine_status", "running": True})

    while not (stop_event and stop_event.is_set()):
        frame = get_frame()
        if frame is None:
            continue

        if mirror:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 1. Facial expressions
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

        # 2. Gestures & body actions
        raw_gest,   gest_conf   = detectGesture(frame)
        raw_action, action_conf = detectBodyAction(frame)

        # 3. Feed processor
        processExpression(raw_expr,   expr_conf)
        processGesture(raw_gest,      gest_conf)
        processBodyAction(raw_action, action_conf)

        # 4. Drain transcript queue each frame
        if audio_ok:
            accumulated_transcripts.extend(_stt.drain())

        # 5. Flush every N seconds
        captions_for_flush = accumulated_transcripts if accumulated_transcripts else None
        stable_results = flushAll(captions=captions_for_flush)

        if stable_results:
            expr, gest, act, sentiment, sent_conf, description = stable_results
            display_expr, display_gest, display_act = expr, gest, act

            accumulated_transcripts.clear()

            lang        = _config.load().get("language", "en")
            description = translate(description, lang)

            broadcast({
                "type":          "result",
                "expression":    expr,
                "gesture":       gest,
                "action":        act,
                "sentiment":     sentiment,
                "sentimentConf": sent_conf,
                "description":   description,
                "summary":       description,
                "language":      lang,          
                "rtl":           is_rtl(lang), 
            })

            if callback:
                callback(expr, gest, act, description)

            threading.Thread(target=speak, args=(description, lang), daemon=True).start()

        # 6. Visual overlay
        audio_label = "audio ✓" if audio_ok else "audio ✗"
        overlay_lines = [
            f"Expression:  {display_expr}",
            f"Gesture:     {display_gest}",
            f"Action:      {display_act}",
            f"Sentiment:   {sentiment} ({sent_conf:.0%})",
            f"Description: {description}",
            f"WS: {len(_ws_clients)}  |  {audio_label}",
        ]

        for i, line in enumerate(overlay_lines):
            cv2.putText(frame, line, (20, h - 30 - (i * 35)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if not headless:
            cv2.imshow("SOVA", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ── Cleanup ───────────────────────────────
    _audio.stop()
    _stt.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_system(source="screen")