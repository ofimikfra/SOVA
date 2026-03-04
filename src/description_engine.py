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
    "You are describing a person's state during a video call. "
    "Write exactly ONE short, human-like, casual sentence, maximum 10 words. "
    "Don't make the sentence complicated or dramatic."
    "Use simple language."
    "Never use first-person ('I', 'I'm', 'I think', 'I can't'). "
    # "Never say 'but', 'however', 'although', 'though', 'genuinely', 'clearly', 'obviously'. "
    "Never trail off or explain your uncertainty. "
    "Never contradict yourself in the same sentence. "
    "End cleanly with a single period. "
    "Use only third-person: 'The person', 'They', 'Their'."
)

_CONFIDENCE_INSTRUCTION = {
    "low": (
        "You MUST use uncertain words: "
        "'seems', 'appears', 'might be', 'looks like'. "
        "NEVER use certain words like 'is', 'looks happy', 'clearly', 'genuinely'. "
        "DO NOT say that the confidence level is low."
    ),
    "medium": (
        "Use hedged language only: "
        "'appears to be', 'seems to be', 'looks like they'. "
        "NEVER use 'is' as a certainty. "
        "DO NOT say that the confidence level is low."
    ),
    "high": (
        "Be direct, no hedging words. "
    ),
}


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(expression: str, gesture: str, action: str,
                  nlp_label: str | None, nlp_conf: float | None,
                  overall_conf: float) -> str:

    tier             = _confidence_tier(overall_conf)
    conf_instruction = _CONFIDENCE_INSTRUCTION[tier]

    gesture_line = f"Gesture: {gesture}" if gesture != "No Gesture" else ""
    action_line  = f"Body:    {action}"  if action  != "Person Center" else ""

    # Only include speech if captions were actually present
    speech_line   = ""
    conflict_line = ""
    if nlp_label is not None:
        speech_line = f"Speech sentiment: {nlp_label} ({nlp_conf:.0%} confidence)"

        conflict = (
            (expression == "Smiling"  and nlp_label == "negative") or
            (expression == "Frowning" and nlp_label == "positive")
        )
        if conflict:
            conflict_line = (
                "Note: expression and speech conflict — "
                "consider sarcasm or mixed feelings."
            )

    lines = filter(None, [
        f"Expression: {expression}",
        gesture_line,
        action_line,
        speech_line,
        conflict_line,
        f"Confidence: {overall_conf:.0%}",
    ])

    return (
        f"{_SYSTEM}\n"
        f"{conf_instruction}\n\n"
        f"Observed signals:\n"
        + "\n".join(lines)
        + "\n\nWrite one sentence only. No 'but', no 'I', no trailing thoughts.\n"
        + "Description:"
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
                "temperature": 0.3,   # lower = more rule-following
                "num_predict": 25,    # 12 words ≈ 16 tokens, 25 gives a little room
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

        # Catch contradictions — e.g. "happy with a frown"
        _CONTRADICTIONS = [
            ("happy", "frown"), ("happy", "concerned"), ("happy", "worried"),
            ("smiling", "frown"), ("positive", "confused"),
        ]
        lower = text.lower()
        for (a, b) in _CONTRADICTIONS:
            if a in lower and b in lower:
                text = ""   # discard — fall through to template
                break

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

def summarize(expression: str, gesture: str, action: str,
              nlp_label: str | None, nlp_conf: float | None,
              overall_conf: float) -> str:
    prompt = _build_prompt(expression, gesture, action,
                           nlp_label, nlp_conf, overall_conf)
    result = _call_ollama(prompt)

    if result:
        print(f"[DESCRIPTION] Ollama ({_confidence_tier(overall_conf)}): {result}")
        return result

    # Template fallback — derive sentiment from what we have
    if nlp_label is not None:
        from src.processor import _fuse_sentiment
        sentiment, _ = _fuse_sentiment(expression, nlp_label, nlp_conf)
    else:
        # No captions — use expression polarity directly
        from src.processor import _EXPR_POLARITY
        polarity, _ = _EXPR_POLARITY.get(expression, (0.0, 0.0))
        if polarity > 0.25:
            sentiment = "positive"
        elif polarity < -0.25:
            sentiment = "negative"
        else:
            sentiment = "neutral"

    result = _template_fallback(expression, gesture, action, sentiment, overall_conf)
    print(f"[DESCRIPTION] Template ({_confidence_tier(overall_conf)}): {result}")
    return result