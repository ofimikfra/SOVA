"""
tests/test_ws_client.py
Simulates the Chrome extension — connects to the running SOVA WebSocket
server, sends fake captions, and prints every result it receives.

Usage:
  1. Start SOVA:             python src/tray_app.py   (or python main.py)
  2. Run this in another terminal: python tests/test_ws_client.py
"""

import asyncio
import json
import sys

WS_URL = "ws://localhost:8765"

FAKE_CAPTIONS = [
    "I think this is a really great idea, I'm excited to move forward.",
    "Yes, that makes total sense to me.",
    "I'm not sure I agree with that approach actually.",
    "Could we look at the data again before deciding?",
    "Absolutely, let's go with that plan.",
]


async def run():
    try:
        import websockets
    except ImportError:
        print("Install websockets:  pip install websockets")
        sys.exit(1)

    print(f"[TEST] Connecting to {WS_URL}...")

    try:
        async with websockets.connect(WS_URL) as ws:
            print("[TEST] Connected ✓\n")

            # Request config immediately like the dashboard does
            await ws.send(json.dumps({"type": "get_config"}))

            caption_idx = 0

            async def send_captions():
                nonlocal caption_idx
                while True:
                    await asyncio.sleep(2)
                    caption = FAKE_CAPTIONS[caption_idx % len(FAKE_CAPTIONS)]
                    caption_idx += 1
                    payload = json.dumps({"type": "caption", "text": caption})
                    await ws.send(payload)
                    print(f"[TEST] Sent caption: \"{caption}\"")

            async def receive():
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        if msg["type"] == "config":
                            print(f"[TEST] Config received: {msg}")
                        elif msg["type"] == "result":
                            print("\n── Result ──────────────────────────────")
                            print(f"  Expression : {msg.get('expression')}")
                            print(f"  Gesture    : {msg.get('gesture')}")
                            print(f"  Action     : {msg.get('action')}")
                            print(f"  Sentiment  : {msg.get('sentiment')} ({msg.get('sentimentConf', 0):.0%})")
                            print(f"  Description: {msg.get('summary')}")
                            print(f"  Dashboard  : {'visible' if msg.get('dashboardVisible') else 'hidden'}")
                            print("────────────────────────────────────────\n")
                    except Exception as e:
                        print(f"[TEST] Parse error: {e}")

            await asyncio.gather(send_captions(), receive())

    except OSError:
        print("[TEST] ✗ Could not connect. Is SOVA running?")
        print("         Start it with:  python main.py")


if __name__ == "__main__":
    asyncio.run(run())