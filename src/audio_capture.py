"""
src/audio_capture.py

Captures system audio and feeds raw numpy float32 chunks (16 kHz mono)
into a queue for the transcription engine.

Strategy by platform / OS version:
  macOS 13+  ->  Swift helper using ScreenCaptureKit (no setup required,
                 one-time permission prompt identical to screen recording)
  macOS 12-  ->  BlackHole virtual audio device (user prompted if missing)
  Windows    ->  WASAPI loopback (built-in, zero setup)
  Linux      ->  PulseAudio/PipeWire monitor device (zero setup)
"""

import sys
import queue
import struct
import threading
import subprocess
import platform
import numpy as np

SAMPLE_RATE = 16000   # Whisper expects 16 kHz
CHUNK_SEC   = 5       # seconds per chunk -- should match flush interval

_audio_queue: queue.Queue = queue.Queue()
_stop_event   = threading.Event()
_capture_thread: threading.Thread | None = None
_helper_proc:    subprocess.Popen | None = None
_current_rms: float = 0.0   # updated each chunk, read by tts_engine


# -- macOS version check -------------------------------------------------------

def _macos_version() -> tuple[int, int]:
    """Returns (major, minor) macOS version, e.g. (13, 4)."""
    ver = platform.mac_ver()[0]
    if not ver:
        return (0, 0)
    parts = ver.split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return (0, 0)


# -- ScreenCaptureKit path (macOS 13+) ----------------------------------------

def _find_swift_helper() -> str | None:
    """Return the path to the compiled binary, or None if not found."""
    import os
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "audio_helper"),
        os.path.join(os.path.dirname(__file__), "audio_helper"),
        os.path.join(getattr(sys, "_MEIPASS", ""), "audio_helper"),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _find_swift_source() -> str | None:
    """Return the path to sova_audio_helper.swift, or None if not found."""
    import os
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "helpers", "audio_helper.swift"),
        os.path.join(os.path.dirname(__file__), "..", "audio_helper.swift"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _ensure_swift_helper() -> str | None:
    """
    Ensure the compiled Swift helper exists.
    If missing, compile it automatically from source — takes ~5 seconds,
    only happens once (or after a clean build).
    Returns the path to the binary, or None on failure.
    """
    import os

    # Already compiled — nothing to do
    existing = _find_swift_helper()
    if existing:
        return existing

    # Find source
    source = _find_swift_source()
    if source is None:
        print(
            "[AUDIO] sova_audio_helper.swift not found.\n"
            "        Make sure helpers/sova_audio_helper.swift is in the project."
        )
        return None

    # Choose output path next to the project root
    output = os.path.normpath(
        os.path.join(os.path.dirname(source), "..", "audio_helper")
    )

    # Check swiftc is available
    swiftc = subprocess.run(["which", "swiftc"], capture_output=True, text=True)
    if swiftc.returncode != 0:
        print(
            "[AUDIO] swiftc not found — Xcode Command Line Tools required.\n"
            "        Install with: xcode-select --install\n"
            "        Then relaunch SOVA."
        )
        return None

    print("[AUDIO] Compiling audio helper (first run only, ~5 seconds)...")

    arch = platform.machine()   # 'arm64' on Apple Silicon, 'x86_64' on Intel

    try:
        # Compile for the current architecture
        result = subprocess.run(
            [
                "swiftc", source,
                "-o", output,
                "-framework", "ScreenCaptureKit",
                "-framework", "CoreAudio",
                "-framework", "AVFoundation",
                f"-target", f"{arch}-apple-macos13.0",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            print(f"[AUDIO] Compilation failed:\n{result.stderr}")
            return None

        os.chmod(output, 0o755)
        print(f"[AUDIO] Audio helper compiled successfully.")
        return output

    except subprocess.TimeoutExpired:
        print("[AUDIO] Compilation timed out.")
        return None
    except Exception as e:
        print(f"[AUDIO] Compilation error: {e}")
        return None


def _screencapturekit_loop():
    """
    Launches the Swift helper, reads chunked PCM from its stdout,
    and puts numpy arrays into _audio_queue.

    Wire protocol (Swift -> Python):
      [4 bytes LE uint32 = byte_count][byte_count bytes of float32 PCM]
    """
    global _helper_proc

    helper = _ensure_swift_helper()
    if helper is None:
        print("[AUDIO] Could not build or locate sova_audio_helper — audio unavailable.")
        return

    try:
        _helper_proc = subprocess.Popen(
            [helper],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as e:
        print(f"[AUDIO] Could not launch Swift helper: {e}")
        return

    print("[AUDIO] ScreenCaptureKit helper started (macOS 13+)")

    def _log_stderr():
        for line in _helper_proc.stderr:
            print("[AUDIO helper]", line.decode(errors="replace").rstrip())

    threading.Thread(target=_log_stderr, daemon=True).start()

    stdout = _helper_proc.stdout

    while not _stop_event.is_set():
        raw_len = stdout.read(4)
        if len(raw_len) < 4:
            break

        byte_count = struct.unpack("<I", raw_len)[0]
        if byte_count == 0:
            continue

        raw_pcm = b""
        remaining = byte_count
        while remaining > 0 and not _stop_event.is_set():
            chunk = stdout.read(remaining)
            if not chunk:
                break
            raw_pcm += chunk
            remaining -= len(chunk)

        if len(raw_pcm) < byte_count:
            break

        samples = np.frombuffer(raw_pcm, dtype=np.float32).copy()

        rms = float(np.sqrt(np.mean(samples ** 2)))
        _current_rms = rms  # always update so duck mode has a live reading

        if rms < 0.001:
            continue

        # Skip transcription queue while TTS is playing, but keep _current_rms
        # updated above so duck mode can still read the real audio level.
        try:
            from src import tts_engine
            if tts_engine.is_tts_active():
                continue
        except Exception:
            pass

        _audio_queue.put(samples)

    print("[AUDIO] ScreenCaptureKit capture thread stopped.")


def _stop_screencapturekit():
    global _helper_proc
    if _helper_proc and _helper_proc.poll() is None:
        try:
            _helper_proc.stdin.close()
            _helper_proc.wait(timeout=3)
        except Exception:
            _helper_proc.terminate()
    _helper_proc = None


# -- sounddevice path (BlackHole / WASAPI / PulseAudio) -----------------------

def _find_loopback_device_sd() -> int | None:
    try:
        import sounddevice as sd
    except ImportError:
        print("[AUDIO] sounddevice not installed -- run: pip install sounddevice")
        return None

    devices = sd.query_devices()

    if sys.platform == "darwin":
        keywords = ("blackhole", "loopback", "soundflower")
        for i, dev in enumerate(devices):
            if (any(k in dev["name"].lower() for k in keywords)
                    and dev["max_input_channels"] > 0):
                print(f"[AUDIO] BlackHole found: '{dev['name']}' (index {i})")
                return i
        return None

    elif sys.platform == "win32":
        for i, dev in enumerate(devices):
            if "loopback" in dev["name"].lower() and dev["max_input_channels"] > 0:
                return i
        try:
            return sd.default.device[1]
        except Exception:
            return None

    else:
        for i, dev in enumerate(devices):
            name = dev["name"].lower()
            if any(k in name for k in ("monitor", "loopback")) \
                    and dev["max_input_channels"] > 0:
                return i
        return None


def _sounddevice_loop(device_index: int, use_wasapi_loopback: bool = False):
    try:
        import sounddevice as sd
    except ImportError:
        return

    frames_per_chunk = int(SAMPLE_RATE * CHUNK_SEC)
    extra_kwargs = {}
    if use_wasapi_loopback and sys.platform == "win32":
        extra_kwargs["extra_settings"] = sd.WasapiSettings(loopback=True)

    print(f"[AUDIO] Capturing via sounddevice (device {device_index})...")

    while not _stop_event.is_set():
        try:
            audio, _ = sd.rec(
                frames_per_chunk,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=device_index,
                **extra_kwargs,
            )
            sd.wait()
            if _stop_event.is_set():
                break
            samples = audio.flatten()
            rms = float(np.sqrt(np.mean(samples ** 2)))
            _current_rms = rms  # always update so duck mode has a live reading
            if rms < 0.001:
                continue
            # Skip transcription queue while TTS is playing, but keep _current_rms
            # updated above so duck mode can still read the real audio level.
            try:
                from src import tts_engine
                if tts_engine.is_tts_active():
                    continue
            except Exception:
                pass
            _audio_queue.put(samples)
        except Exception as e:
            print(f"[AUDIO] sounddevice error: {e}")
            break

    print("[AUDIO] sounddevice capture thread stopped.")


# -- BlackHole install prompt --------------------------------------------------

def _prompt_blackhole_install():
    print(
        "\n"
        "=====================================================\n"
        "  SOVA -- System Audio Setup Required               \n"
        "=====================================================\n"
        "  Your Mac (macOS 12 or earlier) needs a free       \n"
        "  virtual audio driver to capture system audio.     \n"
        "                                                     \n"
        "  1. Download BlackHole 2ch (free):                  \n"
        "     https://existential.audio/blackhole/            \n"
        "                                                     \n"
        "  2. Install it and restart your Mac.                \n"
        "                                                     \n"
        "  3. In System Preferences -> Sound -> Output,       \n"
        "     select BlackHole 2ch.                           \n"
        "                                                     \n"
        "  SOVA will detect it automatically on next launch.  \n"
        "                                                     \n"
        "  Running without audio for now -- sentiment uses    \n"
        "  facial expressions only until setup is complete.   \n"
        "=====================================================\n"
    )
    try:
        import webbrowser
        webbrowser.open("https://existential.audio/blackhole/")
    except Exception:
        pass


# -- Public API ----------------------------------------------------------------

def start(device_index: int | None = None) -> bool:
    """
    Start capturing system audio in a background thread.
    Returns True if capture started, False if unavailable.
    """
    global _capture_thread
    _stop_event.clear()

    while not _audio_queue.empty():
        try:
            _audio_queue.get_nowait()
        except queue.Empty:
            break

    if sys.platform == "darwin":
        major, _ = _macos_version()

        if major >= 13:
            _capture_thread = threading.Thread(
                target=_screencapturekit_loop,
                daemon=True,
                name="audio-capture-sck",
            )
            _capture_thread.start()
            return True
        else:
            dev = device_index or _find_loopback_device_sd()
            if dev is None:
                _prompt_blackhole_install()
                return False
            _capture_thread = threading.Thread(
                target=_sounddevice_loop,
                args=(dev, False),
                daemon=True,
                name="audio-capture-bh",
            )
            _capture_thread.start()
            return True

    elif sys.platform == "win32":
        dev = device_index or _find_loopback_device_sd()
        if dev is None:
            print("[AUDIO] Could not find a WASAPI loopback device.")
            return False
        _capture_thread = threading.Thread(
            target=_sounddevice_loop,
            args=(dev, True),
            daemon=True,
            name="audio-capture-wasapi",
        )
        _capture_thread.start()
        return True

    else:
        dev = device_index or _find_loopback_device_sd()
        if dev is None:
            print(
                "[AUDIO] No monitor device found on Linux.\n"
                "         Try: pactl load-module module-loopback"
            )
            return False
        _capture_thread = threading.Thread(
            target=_sounddevice_loop,
            args=(dev, False),
            daemon=True,
            name="audio-capture-pulse",
        )
        _capture_thread.start()
        return True


def stop():
    _stop_event.set()
    if sys.platform == "darwin":
        major, _ = _macos_version()
        if major >= 13:
            _stop_screencapturekit()


def get_rms() -> float:
    """Current audio RMS level. >0.02 typically means someone is speaking."""
    return _current_rms


def get_queue() -> queue.Queue:
    return _audio_queue


def list_devices():
    try:
        import sounddevice as sd
        print("\n-- Audio Devices ----------------------------")
        for i, dev in enumerate(sd.query_devices()):
            tags = ""
            if dev["max_input_channels"]  > 0: tags += " [in]"
            if dev["max_output_channels"] > 0: tags += " [out]"
            print(f"  {i:>2}: {dev['name']}{tags}")
        print("---------------------------------------------\n")
    except ImportError:
        print("[AUDIO] sounddevice not installed.")