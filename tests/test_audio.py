"""
tests/test_audio.py

Standalone test for the audio capture → Whisper stt pipeline.
Runs independently of the full SOVA system — no webcam, no MediaPipe,
no WebSocket needed.

Usage:
    python tests/test_audio.py

What it does:
    1. Detects and prints all available audio devices
    2. Starts system audio capture
    3. Transcribes each 5-second chunk with Whisper
    4. Prints the transcript + a basic sentiment label
    5. Runs until you press Ctrl+C

Tip: play something with speech (YouTube, a podcast, a call) before
running so there's actual audio to transcribe.
"""

import sys
import os
import time
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import src.audio_capture        as audio
import src.stt_engine as stt
from src.nlp_engine import analyze as nlp_analyze


# ── Colour helpers ────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

def sentiment_colour(label: str) -> str:
    return {
        "positive": GREEN,
        "negative": RED,
        "neutral":  YELLOW,
    }.get(label, RESET)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}SOVA Audio Pipeline Test{RESET}")
    print("─" * 40)

    # Step 1 — list devices so the user can see what's available
    print(f"\n{CYAN}Available audio devices:{RESET}")
    audio.list_devices()

    # Step 2 — start audio capture
    print(f"{CYAN}Starting audio capture...{RESET}")
    ok = audio.start()

    if not ok:
        print(f"\n{RED}Audio capture failed to start.{RESET}")
        print("Check the setup instructions above, then re-run this script.")
        sys.exit(1)

    # Step 3 — start Whisper stt worker
    print(f"{CYAN}Loading Whisper model (first run downloads ~140 MB)...{RESET}\n")
    stt.start(audio.get_queue())

    print(f"{BOLD}Listening...{RESET}  {DIM}(play some speech, then wait ~5 seconds){RESET}")
    print(f"{DIM}Press Ctrl+C to stop.{RESET}\n")
    print("─" * 40)

    chunk_count      = 0
    transcript_count = 0
    accumulated      = []

    def _shutdown(sig, frame):
        print(f"\n\n{BOLD}── Summary ──────────────────────────────{RESET}")
        print(f"  Audio chunks captured : {chunk_count}")
        print(f"  Transcripts received  : {transcript_count}")
        if accumulated:
            label, conf = nlp_analyze(accumulated)
            colour = sentiment_colour(label)
            print(f"  Overall sentiment     : {colour}{label}{RESET} ({conf:.0%})")
            print(f"\n  Full transcript:\n  {DIM}{' '.join(accumulated)}{RESET}")
        print()
        audio.stop()
        stt.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    # Poll for transcripts
    while True:
        new = stt.drain()

        for text in new:
            transcript_count += 1
            accumulated.append(text)

            label, conf = nlp_analyze([text])
            colour = sentiment_colour(label)

            timestamp = time.strftime("%H:%M:%S")
            print(f"  {DIM}[{timestamp}]{RESET}  {text}")
            print(f"           {colour}{label}{RESET} {DIM}({conf:.0%}){RESET}\n")

        # Count chunks arriving in the audio queue (approximate)
        q = audio.get_queue()
        chunk_count += q.qsize()

        time.sleep(0.5)


if __name__ == "__main__":
    main()