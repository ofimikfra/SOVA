import requests
from src import config as _cfg

OLLAMA_URL = "http://localhost:11434/api/generate"
TIMEOUT_S  = 15.0

# ── Confidence tier — internal only, never shown to the model ─────────────────

def _confidence_tier(conf: float) -> str:
    if conf < 0.65:
        return "low"
    elif conf < 0.85:
        return "medium"
    return "high"


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = (
    "You describe a person on a video call in one short, casual sentence. "
    "Maximum 12 words. "
    "Third-person only — 'The person', 'They', or 'Their'. "
    "Never use: 'I', 'but', 'however', 'although', 'confidence', 'uncertain', 'unsure', 'despite'. "
    "Never mention confidence, certainty, or your own limitations. "
    "Never add anything after the final period."
)

# The model only sees the tone instruction — not the word "confidence"
_TONE = {
    "low":    "Use soft observational words: 'seems', 'looks', 'might be', 'appears'.",
    "medium": "Use mild language: 'appears to be', 'seems to be', 'looks like'.",
    "high":   "Be direct and factual. No hedging words.",
}

_EXAMPLES = {
    "low":    "Example: 'The person seems a little distracted.'",
    "medium": "Example: 'They appear to be engaged and listening.'",
    "high":   "Example: 'The person is smiling and nodding.'",
}


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(expression: str, gesture: str,
                  action: str, sentiment: str, overall_conf: float) -> str:
    tier = _confidence_tier(overall_conf)

    gesture_line = f"Gesture: {gesture}" if gesture != "No Gesture" else ""
    action_line  = f"Body:    {action}"  if action  != "Person Center" else ""

    signals = "\n".join(filter(None, [
        f"Expression: {expression}",
        gesture_line,
        action_line,
        f"Sentiment:  {sentiment}",
    ]))

    return (
        f"{_SYSTEM}\n"
        f"{_TONE[tier]} {_EXAMPLES[tier]}\n\n"
        f"Signals:\n{signals}\n\n"
        f"One sentence. End with a period. Nothing after it.\n"
        f"Description:"
    )


# ── Ollama call ───────────────────────────────────────────────────────────────

_BAD_PHRASES = (
    "i think", "i feel", "i believe", "i'm not", "i cant", "i can't",
    "i notice", "confidence", "uncertain", "not sure",
    "but it", "but their", "but the", "however", "although", "despite", "levels", "level",
)

def _call_ollama(prompt: str) -> str | None:
    model = _cfg.load().get("ollama_model", "llama3.2:3b")
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":  model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,  # low = obedient, less rambling
                    "num_predict": 30,   # 12 words ≈ 18 tokens, 30 is a safe cap
                },
            },
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()

        # Strip label if model echoes it back
        if text.lower().startswith("description:"):
            text = text[len("description:"):].strip()

        # Keep only the first sentence
        if "." in text:
            text = text[:text.index(".") + 1]

        # Discard if any banned phrase survived
        lower = text.lower()
        for phrase in _BAD_PHRASES:
            if phrase in lower:
                print(f"[DESCRIPTION] Rejected — contained '{phrase}': {text}")
                return None  # fall through to template

        return text.strip() if text.strip() else None

    except requests.exceptions.ConnectionError:
        print("[DESCRIPTION] Ollama not running — using template fallback")
    except requests.exceptions.Timeout:
        print("[DESCRIPTION] Ollama timed out — using template fallback")
    except Exception as e:
        print(f"[DESCRIPTION] Ollama error: {e} — using template fallback")
    return None


# ── Template fallback ─────────────────────────────────────────────────────────

_TEMPLATES = {
    ("Smiling",         "positive"): "The person is engaged and happy.",
    ("Smiling",         "neutral"):  "The person is relaxed and at ease.",
    ("Smiling",         "negative"): "The person is smiling but the tone seems tense.",
    ("Frowning",        "negative"): "The person looks concerned or displeased.",
    ("Frowning",        "neutral"):  "The person is deep in thought.",
    ("Eyebrows Raised", "positive"): "The person looks pleasantly surprised.",
    ("Eyebrows Raised", "negative"): "The person seems startled or worried.",
    ("Mouth Open",      "positive"): "The person is surprised or animated.",
    ("Mouth Open",      "negative"): "The person looks shocked or taken aback.",
    ("Left Wink",       "positive"): "The person seems playful and lighthearted.",
    ("Right Wink",      "positive"): "The person seems playful and lighthearted.",
    ("Neutral",         "positive"): "The person is calm and content.",
    ("Neutral",         "neutral"):  "The person seems focused and attentive.",
    ("Neutral",         "negative"): "The person looks a little disengaged.",
}

_GESTURE_SUFFIX = {
    "Thumbs Up":   "and is signalling approval.",
    "Thumbs Down": "and is signalling disapproval.",
    "Waving":      "and is waving.",
    "Hand Raised": "and has their hand raised.",
    "Peace Sign":  "and is making a peace sign.",
    "Pointing":    "and is pointing at something.",
    "OK Sign":     "and is giving an OK sign.",
    "Using Phone": "and is on their phone.",
}

_ACTION_SUFFIX = {
    "Leaving Frame": "They may be stepping away.",
    "Looking Away":  "They seem distracted.",
}

_CONFIDENCE_PREFIX = {
    "low":    "It seems like ",
    "medium": "It appears that ",
    "high":   "",
}

def _template_fallback(expression: str, gesture: str,
                       action: str, sentiment: str,
                       overall_conf: float) -> str:
    base = _TEMPLATES.get(
        (expression, sentiment),
        "the person's state is unclear."
    )

    suffix = ""
    if gesture in _GESTURE_SUFFIX:
        base   = base.rstrip(".")
        suffix = " " + _GESTURE_SUFFIX[gesture]
    elif action in _ACTION_SUFFIX:
        suffix = " " + _ACTION_SUFFIX[action]

    tier   = _confidence_tier(overall_conf)
    prefix = _CONFIDENCE_PREFIX[tier]

    if prefix:
        base = base[0].lower() + base[1:]

    return prefix + base + suffix


# ── Public API ────────────────────────────────────────────────────────────────

def summarize(expression: str, gesture: str,
              action: str, sentiment: str,
              overall_conf: float) -> str:
    prompt = _build_prompt(expression, gesture, action, sentiment, overall_conf)
    result = _call_ollama(prompt)

    if result:
        print(f"[DESCRIPTION] Ollama ({_confidence_tier(overall_conf)}): {result}")
        return result

    result = _template_fallback(expression, gesture, action, sentiment, overall_conf)
    print(f"[DESCRIPTION] Template ({_confidence_tier(overall_conf)}): {result}")
    return result