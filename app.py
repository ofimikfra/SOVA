import webview
import threading
import asyncio
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import websockets

clients = set()
ws_loop = None

async def ws_handler(websocket):
    clients.add(websocket)
    try:
        async for message in websocket:
            try:
                msg = json.loads(message)
                if msg.get("type") == "get_config":
                    await websocket.send(json.dumps({
                        "type":           "config",
                        "tts_enabled":    True,
                        "flush_interval": 15,
                        "ollama_model":   "llama3.2:1b"
                    }))
                if msg.get("type") == "settings":
                    from src.processor import set_interval
                    if "flush_interval" in msg:
                        set_interval(float(msg["flush_interval"]))
            except:
                pass
    finally:
        clients.discard(websocket)

async def start_ws():
    async with websockets.serve(ws_handler, "localhost", 8765):
        await asyncio.Future()

def run_ws():
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)
    ws_loop.run_until_complete(start_ws())

# ── Broadcast — now accepts all 6 values from flushAll ──
def broadcast(expr, gest, act, sentiment, sent_conf, description):
    if not clients or ws_loop is None:
        return
    msg = json.dumps({
        "type":          "result",
        "summary":       description,        # full natural language description
        "sentiment":     sentiment,           # positive / negative / neutral
        "sentimentConf": round(sent_conf, 3), # 0.0 – 1.0
    })
    asyncio.run_coroutine_threadsafe(
        asyncio.gather(*[c.send(msg) for c in list(clients)]),
        ws_loop
    )


# ── PyWebView API ─────────────────────────────
class SovaApi:

    def __init__(self):
        self._thread     = None
        self._stop_event = None
        self._running    = False

    def get_status(self) -> dict:
        return {"running": self._running}

    def set_detect_self(self, value: bool) -> dict:
        from main import set_detect_self
        set_detect_self(value)
        return {"ok": True}

    def start(self) -> dict:
        if self._running:
            return {"ok": False, "reason": "Already running"}

        self._stop_event = threading.Event()
        self._running    = True

        def _run():
            try:
                from main import run_system
                run_system(
                    source     = "screen",   # Google Meet screen capture
                    headless   = True,
                    stop_event = self._stop_event,
                    callback   = broadcast,
                )
            except Exception as e:
                print(f"[APP] run_system error: {e}")
            finally:
                self._running = False
                print("[APP] SOVA engine stopped.")

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        print("[APP] SOVA engine started")
        return {"ok": True}

    def stop(self) -> dict:
        if self._stop_event:
            self._stop_event.set()
        self._running = False
        print("[APP] SOVA engine stopping...")
        return {"ok": True}


# ── Entry point ───────────────────────────────
def main():
    threading.Thread(target=run_ws, daemon=True).start()
    print("[APP] WebSocket server started on ws://localhost:8765")

    dashboard = os.path.join(ROOT, "extension", "dashboard.html")
    api = SovaApi()

    webview.create_window(
        title            = "SOVA",
        url              = f"file://{dashboard}",
        js_api           = api,
        width            = 760,
        height           = 860,
        min_size         = (480, 600),
        background_color = "#0f0f0f",
        easy_drag        = False,
    )

    webview.start(debug=False)


if __name__ == "__main__":
    main()
