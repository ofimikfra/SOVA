"""
src/tts_engine.py

Auto-speaks every description at the configured volume.

macOS:  say -v {voice} -o tmpfile → afplay -v {vol} tmpfile
Win/Linux: pyttsx3 with setProperty('volume', vol) + voice matching by locale
"""

import os
import sys
import queue
import tempfile
import threading
import subprocess
import atexit

from src.translator import get_tts_lang, get_macos_voice

# ── State ──────────────────────────────────────────────────────────────────────

_tts_enabled = True
_tts_volume  = 0.25      # 0.0–1.0, default 25%
_tts_active  = False     # True while audio is playing
_queue: queue.Queue = queue.Queue()


def is_tts_active() -> bool:
    """True while TTS audio is playing — used by audio_capture to suppress
    capture and prevent TTS from feeding back into the sentiment pipeline."""
    return _tts_active

# ── Playback ───────────────────────────────────────────────────────────────────

def _play(text: str, tts_lang: str = "en-US"):
    """Speak text at current volume using the correct voice for tts_lang."""
    global _tts_active
    if not text:
        return
    _tts_active = True
    try:
        vol = round(_tts_volume, 2)

        if sys.platform == "darwin":
            # macOS `say` takes a voice name, not a locale code.
            # get_macos_voice() maps e.g. "ar-SA" → "Laila".
            # The voice must be downloaded in System Settings > Accessibility
            # > Spoken Content for non-English languages.
            voice = get_macos_voice(tts_lang.split("-")[0]
                                    if "-" in tts_lang else tts_lang)
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
                tmp = f.name
            subprocess.run(
                ["say", "-v", voice, "-o", tmp, text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["afplay", "-v", str(vol), tmp],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.unlink(tmp)

        else:
            # Windows / Linux — pyttsx3.
            # Pick the first installed voice whose id contains the language
            # prefix (e.g. "ar", "fr", "es"). Falls back to the system default
            # if no matching voice is installed.
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate",   170)
            engine.setProperty("volume", vol)

            lang_prefix = tts_lang.split("-")[0].lower()   # "ar-SA" → "ar"
            voices      = engine.getProperty("voices")
            match       = next(
                (v for v in voices if lang_prefix in v.id.lower()), None
            )
            if match:
                engine.setProperty("voice", match.id)
            else:
                print(f"[TTS] No installed voice found for '{tts_lang}' — using system default.")

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
        item = _queue.get()
        if item is None:
            _queue.task_done()
            break
        text, tts_lang = item          # unpacked tuple put by speak()
        if _tts_enabled and text:
            _play(text, tts_lang)
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


def speak(text: str, locale: str = "en"):
    """Non-blocking. Always replaces stale queued items."""
    tts_lang = get_tts_lang(locale)
    if not _tts_enabled or not text:
        return
    while not _queue.empty():
        try:
            _queue.get_nowait()
            _queue.task_done()
        except queue.Empty:
            break
    _queue.put((text, tts_lang))      # tuple: worker unpacks both fields