import subprocess
import time
import requests
import sys

OLLAMA_URL    = "http://localhost:11434"
OLLAMA_MODEL  = None   # read from config at call time
STARTUP_WAIT  = 8      # seconds to wait for Ollama to become ready
POLL_INTERVAL = 0.5    # how often to check


def _is_running() -> bool:
    """Check if Ollama is already up and accepting requests."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def _model_is_pulled(model: str) -> bool:
    """Check if the requested model is already downloaded."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        # Match loosely — "llama3.2:3b" matches "llama3.2:3b" exactly
        return any(model in m for m in models)
    except Exception:
        return False


def _pull_model(model: str):
    """Pull a model, streaming progress to stdout."""
    print(f"[OLLAMA] Pulling model '{model}' — this may take a few minutes on first run...")
    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[OLLAMA] Failed to pull model: {e}")
    except FileNotFoundError:
        print("[OLLAMA] 'ollama' command not found. Is Ollama installed?")
        print("         Download it from: https://ollama.com/download")


def ensure_ollama(model: str) -> bool:
    """
    Called once at SOVA startup.
    1. If Ollama isn't running, start it.
    2. If the model isn't pulled, pull it.
    Returns True if Ollama is ready, False if something failed.
    """

    # ── Step 1: Start Ollama if not running ──────────────────────────────
    if _is_running():
        print("[OLLAMA] Already running ✓")
    else:
        print("[OLLAMA] Starting Ollama...")
        try:
            # Start as a background process — detached from SOVA's process
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("[OLLAMA] ✗ Ollama not found. Download: https://ollama.com/download")
            return False

        # Wait until it's ready
        deadline = time.time() + STARTUP_WAIT
        while time.time() < deadline:
            if _is_running():
                print("[OLLAMA] Ready ✓")
                break
            time.sleep(POLL_INTERVAL)
        else:
            print("[OLLAMA] ✗ Timed out waiting for Ollama to start.")
            return False

    # ── Step 2: Pull model if not already downloaded ─────────────────────
    if _model_is_pulled(model):
        print(f"[OLLAMA] Model '{model}' ready ✓")
    else:
        _pull_model(model)
        if not _model_is_pulled(model):
            print(f"[OLLAMA] ✗ Model '{model}' could not be pulled — falling back to templates.")
            return False

    _warmup(model)
    return True


def _warmup(model: str):
    """
    Send a throwaway request so Ollama loads the model into memory
    before the first real flush. Prevents timeout on first inference.
    """
    print(f"[OLLAMA] Warming up model '{model}'...")
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model":  model,
                "prompt": "Hi",
                "stream": False,
                "options": {"num_predict": 1},  # one token — just enough to load
            },
            timeout=60,   # generous — first load can be slow
        )
        if resp.status_code == 200:
            print(f"[OLLAMA] Model warm ✓")
        else:
            print(f"[OLLAMA] Warmup returned {resp.status_code} — may be slow on first call")
    except requests.exceptions.Timeout:
        print("[OLLAMA] Warmup timed out — model may still be loading")
    except Exception as e:
        print(f"[OLLAMA] Warmup error: {e}")