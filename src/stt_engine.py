"""
src/stt_engine.py

Consumes raw audio chunks from audio_capture's queue, transcribes
them with faster-whisper (base model), and puts the resulting text
into a transcript queue for the main loop to drain.

faster-whisper is 2-4x faster than vanilla whisper with identical
accuracy and the same model weights.

Install:
    pip install faster-whisper
"""

import queue
import threading

from faster_whisper import WhisperModel

WHISPER_MODEL   = "base"
WHISPER_DEVICE  = "cpu"       # change to "cuda" if you have a GPU
WHISPER_COMPUTE = "int8"      # int8 is fastest on CPU, no quality loss for speech

_transcript_queue: queue.Queue = queue.Queue()
_stop_event = threading.Event()
_model: WhisperModel | None = None
_worker_thread: threading.Thread | None = None


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model():
    global _model
    if _model is None:
        print(f"[TRANSCRIPTION] Loading Whisper {WHISPER_MODEL} model "
              f"({WHISPER_DEVICE}/{WHISPER_COMPUTE})...")
        _model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
        )
        print("[TRANSCRIPTION] Whisper model ready.")
    return _model


# ── Worker ─────────────────────────────────────────────────────────────────────

# AFTER
def _transcribe_worker(audio_queue: queue.Queue):
    from src.tts_engine import is_tts_active   # import here to avoid circular import

    model = _load_model()
    print("[TRANSCRIPTION] Worker started.")

    while not _stop_event.is_set():
        try:
            chunk = audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # TTS is playing — discard this chunk and keep draining the queue
        # until TTS finishes so we don't transcribe our own voice output
        if is_tts_active():
            print("[TRANSCRIPTION] TTS active — discarding audio chunk.")
            while is_tts_active():
                try:
                    audio_queue.get_nowait()   # drain chunks that piled up during TTS
                except queue.Empty:
                    break
            continue

        try:
            segments, info = model.transcribe(
                chunk,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 500,
                },
            )

            text = " ".join(seg.text.strip() for seg in segments).strip()

            if text:
                print(f"[TRANSCRIPTION] → \"{text}\"")
                _transcript_queue.put(text)

        except Exception as e:
            print(f"[TRANSCRIPTION] Error transcribing chunk: {e}")

    print("[TRANSCRIPTION] Worker stopped.")


# ── Public API ─────────────────────────────────────────────────────────────────

def start(audio_queue: queue.Queue):
    """
    Start the transcription worker, consuming from the given audio_queue.
    Call audio_capture.start() first to populate that queue.
    """
    global _worker_thread
    _stop_event.clear()
    _worker_thread = threading.Thread(
        target=_transcribe_worker,
        args=(audio_queue,),
        daemon=True,
        name="transcription-worker",
    )
    _worker_thread.start()


def stop():
    """Signal the transcription worker to stop."""
    _stop_event.set()


def get_queue() -> queue.Queue:
    """Returns the queue that receives transcribed text strings."""
    return _transcript_queue


def drain() -> list[str]:
    """
    Drain all available transcripts and return them as a list.
    Call this each frame in the main loop, just like the old caption queue.
    """
    texts = []
    while True:
        try:
            texts.append(_transcript_queue.get_nowait())
        except queue.Empty:
            break
    return texts