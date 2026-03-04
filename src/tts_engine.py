import queue
import threading
import subprocess
import sys
import atexit

_queue: queue.Queue = queue.Queue()
_tts_enabled = True


def _worker():
    while True:
        text = _queue.get()
        if text is None:            # shutdown signal
            _queue.task_done()
            break
        if _tts_enabled and text:
            try:
                if sys.platform == "darwin":
                    # macOS — use the built-in say command, no dependencies
                    subprocess.run(
                        ["say", text],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    # Windows / Linux — use pyttsx3
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 170)
                    engine.setProperty('volume', 1.0)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
            except Exception as e:
                print(f"[TTS] Error: {e}")
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


def set_enabled(enabled: bool):
    global _tts_enabled
    _tts_enabled = enabled
    print(f"[TTS] {'Enabled' if enabled else 'Disabled'}")


def speak(text: str):
    """Non-blocking. Always speaks the latest description, drops stale queue."""
    if not _tts_enabled or not text:
        return
    while not _queue.empty():
        try:
            _queue.get_nowait()
            _queue.task_done()
        except queue.Empty:
            break
    _queue.put(text)