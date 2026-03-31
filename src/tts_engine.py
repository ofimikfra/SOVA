"""
src/tts_engine.py

Auto-speaks every description at the configured volume.

macOS:  say -o tmpfile → afplay -v {vol} tmpfile
Win/Linux: pyttsx3 with setProperty('volume', vol)
"""

import os
import sys
import queue
import tempfile
import threading
import subprocess
import atexit

# ── State ──────────────────────────────────────────────────────────────────────

_tts_enabled = True
_tts_volume  = 0.25      # 0.0–1.0, default 25%
_tts_active  = False     # True while audio is playing
_queue: queue.Queue = queue.Queue()


def is_tts_active() -> bool:
    """True while TTS audio is playing — used by audio_capture to suppress
    capture and prevent TTS from feeding back into the sentiment pipeline."""
    return _tts_active


# ── Chime ──────────────────────────────────────────────────────────────────────

def _play_chime():
    """Short notification sound when a new description arrives."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(
                ["afplay", "-v", str(round(_tts_volume * 0.6, 2)),
                 "/System/Library/Sounds/Tink.aiff"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


# ── Playback ───────────────────────────────────────────────────────────────────

def _play(text: str):
    """Speak text at current volume. Sets _tts_active for the duration."""
    global _tts_active
    if not text:
        return
    _tts_active = True
    try:
        vol = round(_tts_volume, 2)
        if sys.platform == "darwin":
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
                tmp = f.name
            subprocess.run(["say", "-o", tmp, text],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["afplay", "-v", str(vol), tmp],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.unlink(tmp)
        else:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate",   170)
            engine.setProperty("volume", vol)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
    except Exception as e:
        print(f"[TTS] Playback error: {e}")
    finally:
        _tts_active = False


def _play_test_tone():
    """Play a short test sound at current volume so the user can
    hear the effect of the slider immediately."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(
                ["afplay", "-v", str(round(_tts_volume, 2)),
                 "/System/Library/Sounds/Ping.aiff"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


# ── Worker ─────────────────────────────────────────────────────────────────────

def _worker():
    while True:
        text = _queue.get()
        if text is None:
            _queue.task_done()
            break
        if _tts_enabled and text:
            _play(text)
        _queue.task_done()


_thread = threading.Thread(target=_worker, daemon=True, name="tts-worker")
_thread.start()


def _shutdown():
    while not _queue.empty():
        try:
            _queue.get_nowait()
            _queue.task_done()
        except queue.Empty:
            break
    _queue.put(None)
    _thread.join(timeout=3)


atexit.register(_shutdown)


# ── Public API ─────────────────────────────────────────────────────────────────

def set_enabled(enabled: bool):
    global _tts_enabled
    _tts_enabled = enabled
    print(f"[TTS] {'Enabled' if enabled else 'Disabled'}")


def set_volume(volume: float, play_test: bool = False):
    """Set volume 0.0–1.0. Pass play_test=True to play a confirmation sound."""
    global _tts_volume
    _tts_volume = max(0.0, min(1.0, volume))
    print(f"[TTS] Volume set to {_tts_volume:.0%}")
    if play_test:
        threading.Thread(target=_play_test_tone, daemon=True).start()


def speak(text: str):
    """Non-blocking. Always replaces stale queued items."""
    if not _tts_enabled or not text:
        return
    threading.Thread(target=_play_chime, daemon=True).start()
    while not _queue.empty():
        try:
            _queue.get_nowait()
            _queue.task_done()
        except queue.Empty:
            break
    _queue.put(text)