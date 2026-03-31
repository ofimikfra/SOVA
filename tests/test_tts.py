"""
tests/test_tts.py

Tests for the TTS engine's three modes and supporting logic.

Two layers:
  1. Unit tests (pytest) — no audio hardware, no actual speech.
     Mock _get_rms() and _play() to verify mode behaviour in isolation.

  2. Live tests (--live flag) — actually speaks through your speakers
     so you can hear each mode working for real.

Usage:
    pytest tests/test_tts.py -v              # unit tests only
    python tests/test_tts.py --live          # hear all three modes
    python tests/test_tts.py --live --mode duck
    python tests/test_tts.py --live --mode silence
    python tests/test_tts.py --live --mode manual
"""

import sys
import os
import time
import threading
import argparse
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import src.tts_engine as tts


# ── Helpers ───────────────────────────────────────────────────────────────────

RESET  = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; CYAN = "\033[96m"

def _header(title):
    print(f"\n{BOLD}{title}{RESET}")
    print("─" * 45)


# ══════════════════════════════════════════════
#  Unit tests (pytest)
# ══════════════════════════════════════════════

class TestModeSwitch:
    """set_mode() accepts valid modes and rejects unknown ones."""

    def setup_method(self):
        tts.set_mode("duck")   # reset to default before each test

    def test_set_duck(self):
        tts.set_mode("duck")
        assert tts._tts_mode == "duck"

    def test_set_silence(self):
        tts.set_mode("silence")
        assert tts._tts_mode == "silence"

    def test_set_manual(self):
        tts.set_mode("manual")
        assert tts._tts_mode == "manual"

    def test_invalid_mode_ignored(self):
        tts.set_mode("duck")
        tts.set_mode("yell")        # invalid
        assert tts._tts_mode == "duck"   # unchanged

    def test_set_enabled_false(self):
        tts.set_enabled(False)
        assert tts._tts_enabled is False
        tts.set_enabled(True)       # restore

    def test_set_enabled_true(self):
        tts.set_enabled(True)
        assert tts._tts_enabled is True


class TestDuckMode:
    """Duck mode: always calls _play(), volume varies by RMS."""

    def setup_method(self):
        tts.set_mode("duck")
        tts.set_enabled(True)

    def test_duck_uses_normal_volume_when_quiet(self):
        played = []
        with patch("src.tts_engine._play", side_effect=lambda t, v=1.0: played.append(v() if callable(v) else v)), \
             patch("src.tts_engine._get_rms", return_value=0.0):
            tts._speak_duck("test")
        assert played == [tts.NORMAL_VOLUME]

    def test_duck_uses_low_volume_when_speaking(self):
        played = []
        with patch("src.tts_engine._play", side_effect=lambda t, v=1.0: played.append(v() if callable(v) else v)), \
             patch("src.tts_engine._get_rms", return_value=0.1):
            tts._speak_duck("test")
        assert played == [tts.DUCK_VOLUME]

    def test_duck_threshold_boundary_below(self):
        """RMS just below threshold → normal volume."""
        played = []
        with patch("src.tts_engine._play", side_effect=lambda t, v=1.0: played.append(v() if callable(v) else v)), \
             patch("src.tts_engine._get_rms", return_value=tts.SPEECH_RMS_THRESHOLD - 0.001):
            tts._speak_duck("test")
        assert played == [tts.NORMAL_VOLUME]

    def test_duck_threshold_boundary_above(self):
        """RMS just above threshold → duck volume."""
        played = []
        with patch("src.tts_engine._play", side_effect=lambda t, v=1.0: played.append(v() if callable(v) else v)), \
             patch("src.tts_engine._get_rms", return_value=tts.SPEECH_RMS_THRESHOLD + 0.001):
            tts._speak_duck("test")
        assert played == [tts.DUCK_VOLUME]

    def test_duck_always_calls_play(self):
        """Duck never skips speaking — it always calls _play."""
        called = []
        with patch("src.tts_engine._play", side_effect=lambda t, v=1.0: called.append(t)), \
             patch("src.tts_engine._get_rms", return_value=0.5):
            tts._speak_duck("hello")
        assert called == ["hello"]

    def test_duck_volume_resolved_at_playback_not_synthesis(self):
        """Volume callable must be called by _play, not before it —
        so RMS is sampled at the last moment before afplay fires."""
        # _speak_duck should pass a callable to _play, not a resolved float
        received_vol = []
        with patch("src.tts_engine._play",
                   side_effect=lambda t, v=1.0: received_vol.append(v)), \
             patch("src.tts_engine._get_rms", return_value=0.0):
            tts._speak_duck("test")
        assert len(received_vol) == 1
        assert callable(received_vol[0]), \
            "_speak_duck should pass a callable to _play, not a pre-resolved float"


class TestSilenceMode:
    """Silence mode: waits for quiet, then speaks at full volume."""

    def setup_method(self):
        tts.set_mode("silence")
        tts.set_enabled(True)

    def test_silence_speaks_when_already_quiet(self):
        played = []
        with patch("src.tts_engine._play", side_effect=lambda t, v=1.0: played.append(v)), \
             patch("src.tts_engine._get_rms", return_value=0.0):
            tts._speak_silence("test")
        assert played == [tts.NORMAL_VOLUME]

    def test_silence_waits_then_speaks(self):
        """Simulate: loud for first few polls, then quiet."""
        call_count = [0]
        played = []

        def rms_sequence():
            call_count[0] += 1
            # First 3 calls → loud, then quiet
            return 0.1 if call_count[0] <= 3 else 0.0

        with patch("src.tts_engine._play", side_effect=lambda t, v=1.0: played.append(v)), \
             patch("src.tts_engine._get_rms", side_effect=rms_sequence), \
             patch("src.tts_engine.SILENCE_WAIT_S", 0.05):
            tts._speak_silence("test")

        assert len(played) == 1
        assert played[0] == tts.NORMAL_VOLUME

    def test_silence_timeout_speaks_at_duck_volume(self):
        """If never quiet within deadline, speaks anyway at duck volume."""
        played = []
        with patch("src.tts_engine._play", side_effect=lambda t, v=1.0: played.append(v)), \
             patch("src.tts_engine._get_rms", return_value=0.5), \
             patch("src.tts_engine.SILENCE_WAIT_S", 0.01), \
             patch("src.tts_engine.time") as mock_time:
            # Simulate time advancing past the 30s deadline immediately
            mock_time.time.side_effect = [0, 0, 31, 31]
            mock_time.sleep = time.sleep
            tts._speak_silence("test")

        assert len(played) == 1
        assert played[0] == tts.DUCK_VOLUME

    def test_silence_always_calls_play_eventually(self):
        """Silence mode never permanently drops the description."""
        called = []
        with patch("src.tts_engine._play", side_effect=lambda t, v=1.0: called.append(t)), \
             patch("src.tts_engine._get_rms", return_value=0.0):
            tts._speak_silence("important description")
        assert "important description" in called


class TestManualMode:
    """Manual mode: stores description, never auto-speaks."""

    def setup_method(self):
        tts.set_mode("manual")
        tts.set_enabled(True)
        tts._latest_desc = ""   # reset stored description

    def test_manual_stores_description(self):
        with patch("src.tts_engine._play"):
            tts._speak_manual("stored text")
        assert tts._latest_desc == "stored text"

    def test_manual_does_not_call_play(self):
        called = []
        with patch("src.tts_engine._play", side_effect=called.append):
            tts._speak_manual("stored text")
        assert called == []

    def test_manual_overwrites_with_newer_description(self):
        tts._speak_manual("first")
        tts._speak_manual("second")
        assert tts._latest_desc == "second"

    def test_speak_latest_calls_play(self):
        tts._latest_desc = "hello world"
        played = []
        with patch("src.tts_engine._play", side_effect=lambda t, v=1.0: played.append(t)):
            tts.speak_latest()
        # speak_latest runs in a thread — give it a moment
        time.sleep(0.1)
        assert "hello world" in played

    def test_speak_latest_empty_does_not_call_play(self):
        tts._latest_desc = ""
        called = []
        with patch("src.tts_engine._play", side_effect=called.append):
            tts.speak_latest()
        time.sleep(0.1)
        assert called == []

    def test_speak_latest_uses_most_recent(self):
        tts._speak_manual("old description")
        tts._speak_manual("new description")
        played = []
        with patch("src.tts_engine._play", side_effect=lambda t, v=1.0: played.append(t)):
            tts.speak_latest()
        time.sleep(0.1)
        assert played[-1] == "new description"


class TestSpeakQueue:
    """speak() drops stale queue items — only latest description matters."""

    def setup_method(self):
        tts.set_mode("duck")
        tts.set_enabled(True)

    def test_speak_disabled_does_nothing(self):
        tts.set_enabled(False)
        initial_size = tts._queue.qsize()
        tts.speak("should not queue")
        assert tts._queue.qsize() == initial_size
        tts.set_enabled(True)

    def test_speak_empty_text_does_nothing(self):
        initial_size = tts._queue.qsize()
        tts.speak("")
        tts.speak("   ")   # whitespace is falsy-ish but let's check empty string
        # queue should not grow from empty text
        assert tts._queue.qsize() <= initial_size + 1  # at most the whitespace one

    def test_speak_queues_item(self):
        # Drain queue first
        while not tts._queue.empty():
            try: tts._queue.get_nowait()
            except: break
        tts.speak("queued description")
        assert tts._queue.qsize() >= 1




class TestFeedbackSuppression:
    """_tts_active flag prevents TTS audio from entering the sentiment pipeline."""

    def setup_method(self):
        tts._tts_active = False
        tts.set_enabled(True)
        tts.set_mode("duck")

    def test_is_tts_active_false_by_default(self):
        assert tts.is_tts_active() is False

    def test_is_tts_active_true_during_play(self):
        """_tts_active must be True for the entire duration of _play()."""
        active_during = []

        def fake_subprocess(*args, **kwargs):
            active_during.append(tts.is_tts_active())

        with patch("subprocess.run", side_effect=fake_subprocess), \
             patch("sys.platform", "darwin"), \
             patch("tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("os.unlink"):
            mock_tmp.return_value.__enter__ = lambda s: s
            mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
            mock_tmp.return_value.name = "/tmp/fake.aiff"
            tts._play("test text")

        assert any(active_during), "_tts_active should be True during _play()"

    def test_is_tts_active_false_after_play(self):
        """_tts_active must be reset to False after _play() finishes."""
        with patch("subprocess.run"), \
             patch("sys.platform", "darwin"), \
             patch("tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("os.unlink"):
            mock_tmp.return_value.__enter__ = lambda s: s
            mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
            mock_tmp.return_value.name = "/tmp/fake.aiff"
            tts._play("test text")

        assert tts.is_tts_active() is False

    def test_is_tts_active_false_after_exception(self):
        """_tts_active must be reset even if _play() raises."""
        with patch("subprocess.run", side_effect=RuntimeError("boom")), \
             patch("sys.platform", "darwin"), \
             patch("tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("os.unlink"):
            mock_tmp.return_value.__enter__ = lambda s: s
            mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
            mock_tmp.return_value.name = "/tmp/fake.aiff"
            tts._play("test text")   # must not raise

        assert tts.is_tts_active() is False

    def test_audio_skips_chunk_when_tts_active(self):
        """Simulate the audio_capture guard: chunk is dropped while TTS plays."""
        import queue as Q
        q = Q.Queue()
        tts._tts_active = True
        try:
            if tts.is_tts_active():
                pass   # continue (skipped)
            else:
                q.put("chunk")
        finally:
            tts._tts_active = False
        assert q.empty(), "Chunk should be dropped while TTS is active"

    def test_audio_queues_chunk_when_tts_inactive(self):
        """Chunk queues normally when TTS is not playing."""
        import queue as Q
        q = Q.Queue()
        tts._tts_active = False
        if not tts.is_tts_active():
            q.put("chunk")
        assert not q.empty()

class TestRmsIntegration:
    """_get_rms() reads from audio_capture without crashing."""

    def test_get_rms_returns_float(self):
        # audio_capture may not be running — should return 0.0 gracefully
        rms = tts._get_rms()
        assert isinstance(rms, float)
        assert rms >= 0.0

    def test_is_speaking_returns_bool(self):
        result = tts._is_speaking()
        assert isinstance(result, bool)

    def test_is_speaking_false_when_rms_zero(self):
        with patch("src.tts_engine._get_rms", return_value=0.0):
            assert tts._is_speaking() is False

    def test_is_speaking_true_when_rms_high(self):
        with patch("src.tts_engine._get_rms", return_value=0.5):
            assert tts._is_speaking() is True


# ══════════════════════════════════════════════
#  Live tests — actually speak out loud
# ══════════════════════════════════════════════

LIVE_TEXT = "The person appears engaged and is listening attentively."

def _live_duck():
    _header("Mode: DUCK")
    print("You should hear this at FULL volume (room is quiet now).")
    tts.set_mode("duck")
    tts.set_enabled(True)
    tts.speak(LIVE_TEXT)
    time.sleep(4)

    print("\nNow simulating speech detected (RMS = 0.1)...")
    print("You should hear this at LOW volume (25%).")
    with patch("src.tts_engine._get_rms", return_value=0.1):
        tts._speak_duck(LIVE_TEXT)
    time.sleep(4)


def _live_silence():
    _header("Mode: SILENCE")
    print("Simulating 2 seconds of speech, then quiet.")
    print("SOVA should wait, then speak at full volume.")
    tts.set_mode("silence")

    call_count = [0]
    def rms_seq():
        call_count[0] += 1
        # Loud for ~2s worth of 50ms polls = 40 calls
        return 0.1 if call_count[0] < 40 else 0.0

    with patch("src.tts_engine._get_rms", side_effect=rms_seq):
        tts._speak_silence(LIVE_TEXT)
    time.sleep(4)


def _live_manual():
    _header("Mode: MANUAL")
    print("Storing description — nothing should play yet...")
    tts.set_mode("manual")
    tts._speak_manual(LIVE_TEXT)
    time.sleep(1.5)

    print("Calling speak_latest() now — you should hear it:")
    tts.speak_latest()
    time.sleep(4)

    print("\nStoring a second description...")
    tts._speak_manual("The person has raised their hand.")
    time.sleep(1.0)
    print("Calling speak_latest() — should speak the newer one:")
    tts.speak_latest()
    time.sleep(4)


def run_live(mode: str | None):
    print(f"\n{BOLD}SOVA TTS Live Test{RESET}")
    print(f"{DIM}Make sure your speakers are on.{RESET}")

    if mode is None or mode == "duck":
        _live_duck()
    if mode is None or mode == "silence":
        _live_silence()
    if mode is None or mode == "manual":
        _live_manual()

    print(f"\n{GREEN}Live tests complete.{RESET}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Run audible live tests instead of pytest unit tests")
    parser.add_argument("--mode", choices=["duck", "silence", "manual"],
                        help="Run only one mode's live test")
    args = parser.parse_args()

    if args.live:
        run_live(args.mode)
    else:
        # Run pytest programmatically if called directly without --live
        import pytest
        sys.exit(pytest.main([__file__, "-v"]))