"""
tests/test_dashboard.py
────────────────────────────────────────────────────────────────
Lightweight WebSocket server that pumps fake SOVA result messages
to the dashboard so you can test the UI without running the full
SOVA pipeline.

Usage
─────
    # From the project root:
    python tests/test_dashboard.py

Then open extension/dashboard.html in a browser
(or load the extension and open the dashboard tab).

Options
───────
    --interval  N   Seconds between messages  (default: 4)
    --port      N   WebSocket port            (default: 8765)
    --once          Send one burst then wait for Ctrl-C (useful for
                    snapshot testing — doesn't spam the feed)

The server also responds to "get_config" messages from the dashboard
so the settings panel populates correctly.
"""

import asyncio
import json
import random
import argparse
import signal
import sys
import time
from datetime import datetime

try:
    import websockets
except ImportError:
    print("[TEST] websockets not installed — run: pip install websockets")
    sys.exit(1)

# ── Fake data pools ───────────────────────────────────────────────────────────

_EXPRESSIONS = [
    "Smiling", "Neutral", "Frowning",
    "Eyebrows Raised", "Mouth Open", "Left Wink",
]

_GESTURES = [
    "No Gesture", "No Gesture", "No Gesture",   # weighted towards none
    "Thumbs Up", "Waving", "Pointing", "Peace Sign", "Hand Raised",
]

_ACTIONS = [
    "Person Center", "Person Center", "Person Center",  # weighted towards center
    "Looking Away", "Person Left", "Person Right",
]

_SENTIMENTS = ["positive", "negative", "neutral"]

_DESCRIPTIONS = {
    ("Smiling",         "positive"): [
        "The person is engaged and clearly enjoying the conversation.",
        "They look happy and at ease.",
        "The person seems genuinely pleased.",
    ],
    ("Smiling",         "neutral"): [
        "The person is relaxed and attentive.",
        "They appear comfortable and focused.",
    ],
    ("Frowning",        "negative"): [
        "The person looks concerned or displeased.",
        "They seem a little frustrated.",
        "The person appears troubled by something.",
    ],
    ("Frowning",        "neutral"): [
        "The person is deep in thought.",
        "They look like they're concentrating hard.",
    ],
    ("Eyebrows Raised", "positive"): [
        "The person looks pleasantly surprised.",
        "They seem impressed by what they heard.",
    ],
    ("Neutral",         "neutral"): [
        "The person seems focused and attentive.",
        "They are listening carefully.",
        "The person appears calm and composed.",
    ],
    ("Neutral",         "positive"): [
        "The person is calm and content.",
        "They look comfortable and engaged.",
    ],
    ("Mouth Open",      "positive"): [
        "The person is surprised or animated.",
        "They look energised and expressive.",
    ],
}

_FALLBACK_DESCRIPTIONS = [
    "The person seems engaged in the conversation.",
    "They appear to be listening attentively.",
    "The person looks focused.",
    "They seem to be following along closely.",
    "The person appears present and alert.",
]


def _make_result() -> dict:
    expr      = random.choice(_EXPRESSIONS)
    gesture   = random.choice(_GESTURES)
    action    = random.choice(_ACTIONS)
    sentiment = random.choice(_SENTIMENTS)
    conf      = round(random.uniform(0.55, 0.97), 3)

    pool = _DESCRIPTIONS.get((expr, sentiment), _FALLBACK_DESCRIPTIONS)
    desc = random.choice(pool)

    return {
        "type":          "result",
        "expression":    expr,
        "gesture":       gesture,
        "action":        action,
        "sentiment":     sentiment,
        "sentimentConf": conf,
        "description":   desc,
        "summary":       desc,   # dashboard.js and popup.js read "summary"
    }


_MOCK_CONFIG = {
    "type":           "config",
    "ollama_model":   "llama3.2:3b",
    "flush_interval": 30,
    "tts_enabled":    True,
}


# ── Server ────────────────────────────────────────────────────────────────────

_clients: set = set()


async def _handler(websocket, interval: float, once: bool):
    _clients.add(websocket)
    addr = websocket.remote_address
    print(f"[TEST] Dashboard connected  ({addr})  — total: {len(_clients)}")

    # Send config immediately so settings panel populates
    await websocket.send(json.dumps(_MOCK_CONFIG))

    try:
        # Kick off the pump task for this client
        pump = asyncio.create_task(_pump(websocket, interval, once))

        async for raw in websocket:
            try:
                msg = json.loads(raw)
                if msg.get("type") == "get_config":
                    await websocket.send(json.dumps(_MOCK_CONFIG))
                    print(f"[TEST] Sent config to {addr}")
                elif msg.get("type") == "settings":
                    print(f"[TEST] Received settings from dashboard: {msg}")
            except json.JSONDecodeError:
                pass

        pump.cancel()

    except (websockets.exceptions.ConnectionClosedOK,
            websockets.exceptions.ConnectionClosedError):
        pass
    finally:
        _clients.discard(websocket)
        print(f"[TEST] Dashboard disconnected ({addr}) — total: {len(_clients)}")


async def _pump(websocket, interval: float, once: bool):
    """Send result messages on a timer."""
    count = 0
    try:
        while True:
            result = _make_result()
            count += 1
            ts = datetime.now().strftime("%H:%M:%S")
            print(
                f"[TEST] [{ts}] #{count:>3}  "
                f"expr={result['expression']:<16}  "
                f"sentiment={result['sentiment']:<8}  "
                f"conf={result['sentimentConf']:.0%}  "
                f"→ \"{result['summary']}\""
            )
            await websocket.send(json.dumps(result))

            if once and count >= 5:
                print("[TEST] --once: burst complete, holding connection open.")
                await asyncio.sleep(9999)

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


async def _serve(port: int, interval: float, once: bool):
    handler = lambda ws, path=None: _handler(ws, interval, once)
    async with websockets.serve(handler, "localhost", port):
        print(f"[TEST] ╔══════════════════════════════════════╗")
        print(f"[TEST] ║  SOVA Dashboard Test Server          ║")
        print(f"[TEST] ║  ws://localhost:{port:<5}                 ║")
        print(f"[TEST] ║  Interval: {interval}s                      ║")
        print(f"[TEST] ╚══════════════════════════════════════╝")
        print(f"[TEST]")
        print(f"[TEST] Open extension/dashboard.html in a browser,")
        print(f"[TEST] or load the Chrome extension and open the dashboard.")
        print(f"[TEST] Ctrl-C to stop.\n")
        await asyncio.Future()  # run forever


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Feed fake SOVA results to the dashboard for UI testing."
    )
    parser.add_argument(
        "--interval", type=float, default=4.0,
        metavar="N",
        help="Seconds between result messages (default: 4)",
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        metavar="N",
        help="WebSocket port (default: 8765)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Send a burst of 5 messages then hold the connection open",
    )
    args = parser.parse_args()

    # Graceful Ctrl-C
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _stop(sig, frame):
        print("\n[TEST] Shutting down.")
        loop.stop()

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        loop.run_until_complete(_serve(args.port, args.interval, args.once))
    except RuntimeError:
        pass   # loop stopped cleanly


if __name__ == "__main__":
    main()