"""
tests/test_transcript_nlp.py

Tests that transcripts from the audio pipeline flow correctly into
the NLP engine and produce sensible sentiment labels.

Two modes:
  1. Live mode  — captures real audio, transcribes it, runs NLP on each
                  chunk as it arrives. (default)
  2. Replay mode — feeds a list of hardcoded strings through NLP only,
                  no audio hardware needed. Run with --replay.

Usage:
    python tests/test_transcript_nlp.py           # live
    python tests/test_transcript_nlp.py --replay  # offline
"""

import sys
import os
import time
import signal
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.nlp_engine import analyze as nlp_analyze
import src.audio_capture        as audio
import src.stt_engine as stt


# ── Colours ───────────────────────────────────────────────────────────────────

RESET  = "\033[0m";  BOLD  = "\033[1m";  DIM   = "\033[2m"
GREEN  = "\033[92m"; YELLOW= "\033[93m"; RED   = "\033[91m"; CYAN = "\033[96m"

def _col(label):
    return {"positive": GREEN, "negative": RED, "neutral": YELLOW}.get(label, RESET)


# ── Shared display ────────────────────────────────────────────────────────────

def _show(transcript: str, label: str, conf: float, source: str = ""):
    ts  = time.strftime("%H:%M:%S")
    tag = f"  {DIM}[{ts}]{RESET}"
    if source:
        tag += f" {DIM}({source}){RESET}"
    print(f"{tag}  \"{transcript}\"")
    print(f"           {_col(label)}{label}{RESET}  {DIM}{conf:.0%}{RESET}\n")


def _summary(all_transcripts: list[str]):
    print(f"\n{BOLD}── Summary ──────────────────────────────{RESET}")
    if not all_transcripts:
        print(f"  {DIM}No transcripts received.{RESET}")
        return
    label, conf = nlp_analyze(all_transcripts)
    print(f"  Transcripts      : {len(all_transcripts)}")
    print(f"  Overall sentiment: {_col(label)}{label}{RESET}  {DIM}({conf:.0%} confidence){RESET}")
    print(f"\n  Full transcript:")
    print(f"  {DIM}{' '.join(all_transcripts)}{RESET}\n")


# ── Mode 1: Live ──────────────────────────────────────────────────────────────

def run_live():
    print(f"\n{BOLD}Transcript → NLP  [live mode]{RESET}")
    print(f"{DIM}Audio is captured, transcribed by Whisper, then scored by DistilBERT.{RESET}")
    print("─" * 50)

    print(f"\n{CYAN}Starting audio capture...{RESET}")
    ok = audio.start()
    if not ok:
        print(f"{RED}Audio capture failed — see setup instructions above.{RESET}")
        sys.exit(1)

    print(f"{CYAN}Loading Whisper model...{RESET}\n")
    stt.start(audio.get_queue())

    print(f"{BOLD}Listening...{RESET}  {DIM}speak or play audio, wait ~5s per chunk{RESET}")
    print(f"{DIM}Ctrl+C to stop and see summary.{RESET}\n")
    print("─" * 50)

    all_transcripts = []

    def _shutdown(sig, frame):
        _summary(all_transcripts)
        audio.stop()
        stt.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    while True:
        for text in stt.drain():
            label, conf = nlp_analyze([text])
            all_transcripts.append(text)
            _show(text, label, conf, source="whisper→nlp")
        time.sleep(0.3)


# ── Mode 2: Replay ────────────────────────────────────────────────────────────

REPLAY_TRANSCRIPTS = [
    # Positive
    "This is a great idea, I think we should go with it.",
    "Absolutely, I completely agree with you on that.",
    "The results look fantastic, really well done everyone.",
    # Neutral / ambiguous
    "Let's move on to the next agenda item.",
    "I'm not sure, we might need more time to look at this.",
    "Okay, so can someone take notes on this section?",
    # Negative
    "I really disagree, this approach has too many problems.",
    "This isn't working and I'm quite frustrated with the progress.",
    "Honestly I think we've wasted a lot of time on this.",
]

def run_replay():
    print(f"\n{BOLD}Transcript → NLP  [replay mode]{RESET}")
    print(f"{DIM}Feeds hardcoded strings through the NLP engine only — no audio hardware needed.{RESET}")
    print("─" * 50 + "\n")

    all_transcripts = []

    for text in REPLAY_TRANSCRIPTS:
        label, conf = nlp_analyze([text])
        all_transcripts.append(text)
        _show(text, label, conf, source="replay")
        time.sleep(0.2)   # small pause so output is readable

    # Also test cumulative — simulates what flushAll sees
    print(f"{BOLD}Cumulative NLP (all chunks together):{RESET}")
    label, conf = nlp_analyze(all_transcripts)
    print(f"  {_col(label)}{label}{RESET}  {DIM}({conf:.0%}){RESET}\n")

    _summary(all_transcripts)

    # Verify a few known expectations
    print(f"{BOLD}Assertions:{RESET}")
    _assert_nlp("This is a great idea, I love it!", "positive")
    _assert_nlp("I really disagree, this is terrible.", "negative")
    _assert_nlp("", "neutral")
    _assert_nlp("   ", "neutral")
    print(f"\n{GREEN}All assertions passed.{RESET}\n")


def _assert_nlp(text: str, expected: str):
    label, conf = nlp_analyze([text] if text.strip() else [])
    status = f"{GREEN}✓{RESET}" if label == expected else f"{RED}✗{RESET}"
    print(f"  {status}  \"{text[:40]}\"  →  {_col(label)}{label}{RESET}"
          f"  {DIM}(expected {expected}){RESET}")
    if label != expected:
        print(f"     {RED}FAIL: got '{label}' ({conf:.0%}), expected '{expected}'{RESET}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Run in offline replay mode (no audio hardware needed)",
    )
    args = parser.parse_args()

    if args.replay:
        run_replay()
    else:
        run_live()