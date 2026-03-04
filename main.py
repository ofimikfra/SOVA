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
            except json.JSONDecodeError:
                pass
    except (websockets.exceptions.ConnectionClosedOK,
            websockets.exceptions.ConnectionClosedError):
        pass
    finally:
        _ws_clients.discard(websocket)
        print(f"[WS] Extension disconnected — {len(_ws_clients)} client(s))")


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
    """Schedule a broadcast from any thread."""
    if _ws_loop and _ws_clients:
        asyncio.run_coroutine_threadsafe(_ws_broadcast(payload), _ws_loop)


async def _ws_serve():
    async with websockets.serve(_ws_handler, WS_HOST, WS_PORT):
        print(f"[WS] Server listening on ws://{WS_HOST}:{WS_PORT}")
        await asyncio.get_event_loop().create_future()  # run forever


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

    ws_thread = threading.Thread(target=_start_ws_thread, daemon=True)
    ws_thread.start()
    time.sleep(0.5)  # Give server time to bind

    if source == "screen":
        get_frame = getScreenFrame
        mirror    = False
        print("[SOVA] Monitoring Screen...")
    else:
        get_frame = getCameraFrame
        mirror    = True
        print("[SOVA] Monitoring Webcam...")

    display_expr      = "Neutral"
    display_gest      = "No Gesture"
    display_act       = "Person Center"

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
        
        # 5. Flush every 5 s
        stable_results = flushAll(captions=pending_captions)   # ← pass captions in
        if stable_results:
            expr, gest, act, sentiment, sent_conf, description = stable_results  # ← unpack values
        
            display_expr, display_gest, display_act = expr, gest, act
    
        
            _broadcast_sync({
                "type":             "result",
                "expression":       expr,
                "gesture":          gest,
                "action":           act,
                "sentiment":        sentiment,
                "sentimentConf":    sent_conf,
                "description":      description,  
            })

            if callback:
                callback(expr, gest, act, f"Detected {expr}")

            threading.Thread(target=speak, args=(description,), daemon=True).start()

        # 6. Visual Overlay
        overlay_lines = [
            f"Expression:  {display_expr}",
            f"Gesture:     {display_gest}",
            f"Action:      {display_act}",
            f"Sentiment:   {sentiment} ({sent_conf:.0%})",
            f"Description: {description}",
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