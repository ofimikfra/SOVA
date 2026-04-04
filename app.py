"""
app.py — SOVA Desktop App (debug build)
"""
import webview
import threading
import asyncio
import json
import os
import sys
import traceback

try:
    import websockets
except ImportError:
    websockets = None

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

WS_HOST = "localhost"
WS_PORT = 8765

_ws_clients: set = set()
_ws_loop: asyncio.AbstractEventLoop | None = None
_sova_api = None
_webview_window = None


# ── Push directly into PyWebView via evaluate_js ──────────────────────────────

def _push_to_webview(payload: dict):
    if _webview_window is None:
        return
    try:
        js = f"window.__sovaReceive && window.__sovaReceive({json.dumps(payload)})"
        _webview_window.evaluate_js(js)
    except Exception as e:
        print(f"[APP] evaluate_js error: {e}")


# ── WS broadcast (Chrome extension only) ─────────────────────────────────────

async def _ws_broadcast(payload: dict):
    if not _ws_clients:
        return
    msg = json.dumps(payload)
    dead = set()
    for ws in _ws_clients.copy():
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


def ws_broadcast_sync(payload: dict):
    _push_to_webview(payload)
    if _ws_loop and _ws_clients:
        asyncio.run_coroutine_threadsafe(_ws_broadcast(payload), _ws_loop)


# ── WS server ─────────────────────────────────────────────────────────────────

async def _ws_handler(websocket):
    req = getattr(websocket, 'request', None) or getattr(websocket, 'request_headers', None)
    if req is not None:
        hdrs = getattr(req, 'headers', req)
        origin = hdrs.get('Origin', hdrs.get('origin', 'unknown'))
        host   = hdrs.get('Host',   hdrs.get('host',   'unknown'))
        path   = getattr(req, 'path', getattr(websocket, 'path', '/'))
    else:
        origin = getattr(websocket, 'origin', 'unknown')
        host   = 'unknown'
        path   = getattr(websocket, 'path', '/')

    _ws_clients.add(websocket)
    print(f"[APP WS] *** Client {len(_ws_clients)} connected ***")
    print(f"[APP WS]     path={path}  origin={origin}  host={host}")
    print(f"[APP WS]     remote={websocket.remote_address}")
    print(f"[APP WS]     total clients now: {len(_ws_clients)}")

    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
                t   = msg.get("type")

                print(f"[APP WS]   msg from {websocket.remote_address}: type={t}")

                if t == "identify":
                    print(f"[APP WS]   ^^^ identifies as: {msg.get('name', '?')}")

                elif t == "start_engine":
                    result = _sova_api.start() if _sova_api else {"ok": False}
                    if not result.get("ok", False):
                        # Only broadcast failure — success is broadcast by main.py
                        # once the loop is actually running
                        payload = {"type": "engine_status", "running": False}
                        await _ws_broadcast(payload)
                        _push_to_webview(payload)

                elif t == "stop_engine":
                    if _sova_api: _sova_api.stop()
                    # engine_status: false is broadcast by the thread's finally block
                    # — don't broadcast here to avoid duplicates

                elif t == "get_config":
                    from src import config as _cfg
                    await websocket.send(json.dumps({"type": "config", **_cfg.load()}))

                elif t == "settings":
                    _apply_settings_from_msg(msg)
                    cfg = {"type": "config", **_cfg_load()}
                    await _ws_broadcast(cfg)
                    _push_to_webview(cfg)

            except json.JSONDecodeError:
                pass

    except (websockets.exceptions.ConnectionClosedOK,
            websockets.exceptions.ConnectionClosedError):
        pass
    finally:
        _ws_clients.discard(websocket)
        print(f"[APP WS] Client disconnected: {websocket.remote_address} — {len(_ws_clients)} remaining")


def _cfg_load():
    from src import config as _cfg
    return _cfg.load()


def _apply_settings_from_msg(msg: dict):
    """Apply a settings dict: write to config.json and hot-reload running modules."""
    from src import config as _cfg
    import src.processor as _proc
    import src.tts_engine as _tts

    _cfg.update(msg)
    if "flush_interval" in msg: _proc.set_interval(float(msg["flush_interval"]))
    if "tts_enabled"    in msg: _tts.set_enabled(msg["tts_enabled"])
    if "tts_volume"     in msg: _tts.set_volume(float(msg["tts_volume"]))


async def _ws_serve():
    if websockets is None:
        print("[APP WS] websockets not installed"); return
    async with websockets.serve(_ws_handler, WS_HOST, WS_PORT):
        print(f"[APP WS] Server ready on ws://{WS_HOST}:{WS_PORT}")
        await asyncio.get_event_loop().create_future()


def _start_ws_thread():
    global _ws_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _ws_loop = loop
    loop.run_until_complete(_ws_serve())


# ── SovaApi ───────────────────────────────────────────────────────────────────

class SovaApi:

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._running = False

    def get_status(self) -> dict:
        return {"running": self._running}

    def save_config(self, settings: dict) -> dict:
        """
        Called directly by dashboard.js via pywebview when SOVA is not running.
        Writes to config.json and hot-reloads any running modules immediately.
        """
        _apply_settings_from_msg(settings)
        print(f"[APP] Config saved via pywebview: {settings}")
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
                    source       = "screen",
                    headless     = True,
                    stop_event   = self._stop_event,
                    ws_broadcast = ws_broadcast_sync,
                )
            except Exception as e:
                print(f"[APP] run_system error: {e}")
                traceback.print_exc()
            finally:
                self._running = False
                # Single authoritative engine_status: false — emitted here only,
                # not in stop() — so the dashboard never hears it more than once
                ws_broadcast_sync({"type": "engine_status", "running": False})
                print("[APP] SOVA engine stopped.")

        self._thread = threading.Thread(target=_run, daemon=True, name="sova-engine")
        self._thread.start()
        print("[APP] SOVA engine started")
        return {"ok": True}

    def stop(self) -> dict:
        if self._stop_event: self._stop_event.set()
        self._running = False
        # Don't broadcast here — the thread's finally block handles it once cleanly
        print("[APP] SOVA engine stopping...")
        return {"ok": True}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global _sova_api, _webview_window

    dashboard = os.path.join(ROOT, "extension", "dashboard.html")
    _sova_api = SovaApi()

    ws_thread = threading.Thread(target=_start_ws_thread, daemon=True, name="ws-server")
    ws_thread.start()

    def _preload():
        from src import config       # noqa
        import src.processor         # noqa  triggers: nlp_engine → from transformers import pipeline
        import src.tts_engine        # noqa
    threading.Thread(target=_preload, daemon=True, name="preload").start()

    _webview_window = webview.create_window(
        title            = "SOVA",
        url              = f"file://{dashboard}",
        js_api           = _sova_api,
        width            = 760,
        height           = 860,
        min_size         = (480, 600),
        background_color = "#0f0f0f",
        easy_drag        = False,
    )

    webview.start(debug=False)


if __name__ == "__main__":
    main()