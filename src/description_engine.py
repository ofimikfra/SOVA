"""
SOVA Summary Engine
Generates a human-like description of the user's current state using
a local Ollama model. Falls back to a rule-based template if Ollama
is unavailable.

Requirements:
  1. Install Ollama: https://ollama.com/download
  2. Pull the model: ollama pull llama3.2:3b
  3. Ollama must be running before SOVA starts: ollama serve
"""

import requests

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"   # swap to "mistral" or "llama3.2:1b" if needed
TIMEOUT_S    = 4.0             # max wait before falling back to template

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a concise observer describing a person's current state "
    "during a video call. Write exactly ONE natural sentence (max 20 words). "
    "No lists, no labels, no punctuation beyond the final period. "
    "Sound human and conversational, not clinical."
)

_CONFIDENCE_INSTRUCTION = {
    "low":    "You are NOT confident in these signals. Use uncertain language like "
              "'it seems like', 'possibly', or 'it looks like'.",
    "medium": "You are moderately confident. Use hedged language like "
              "'it appears that' or 'they seem to be'.",
    "high":   "You are fully confident. Speak directly with no hedging.",
}

def _confidence_tier(conf: float) -> str:
    if conf < 0.65:
        return "low"
    elif conf < 0.85:
        return "medium"
    return "high"

def _build_prompt(expression: str, gesture: str,
                  action: str, sentiment: str, overall_conf: float) -> str:
    tier        = _confidence_tier(overall_conf)
    conf_instruction = _CONFIDENCE_INSTRUCTION[tier]

    gesture_line = f"Gesture: {gesture}" if gesture != "No Gesture" else ""
    action_line  = f"Body:    {action}"  if action  != "Person Center" else ""

    lines = filter(None, [
        f"Expression: {expression}",
        gesture_line,
        action_line,
        f"Sentiment:  {sentiment}",
        f"Confidence: {overall_conf:.0%}",
    ])
    return (
        f"{_SYSTEM}\n"
        f"{conf_instruction}\n\n"
        f"Observed signals:\n"
        + "\n".join(lines)
        + "\n\nDescription:"
    )


# ── Ollama call ───────────────────────────────────────────────────────────────

def _call_ollama(prompt: str) -> str | None:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 40,  # cap tokens → keeps output short & fast
                },
            },
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        # Strip any leading label the model might add ("Description: ...")
        if text.lower().startswith("description:"):
            text = text[len("description:"):].strip()
        return text if text else None
    except requests.exceptions.ConnectionError:
        print("[SUMMARY] Ollama not running — using template fallback")
    except requests.exceptions.Timeout:
        print("[SUMMARY] Ollama timed out — using template fallback")
    except Exception as e:
        print(f"[SUMMARY] Ollama error: {e} — using template fallback")
    return None


# ── Template fallback ─────────────────────────────────────────────────────────
# Used when Ollama is unavailable. Covers the most common signal combinations.

_TEMPLATES = {
    ("Smiling",         "positive"): "The person is engaged and happy.",
    ("Smiling",         "neutral"):  "The person is relaxed and at ease.",
    ("Smiling",         "negative"): "The person is smiling but the tone seems tense.",
    ("Frowning",        "negative"): "The person looks concerned or displeased.",
    ("Frowning",        "neutral"):  "The person is deep in thought.",
    ("Eyebrows Raised", "positive"): "The person looks pleasantly surprised.",
    ("Eyebrows Raised", "negative"): "The person seems startled or worried.",
    ("Mouth Open",      "positive"): "The person is animated and engaged.",
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
        "the person's state is unclear."   # lowercase — prefix prepended below
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
        # Lowercase the first letter of base so it flows after the prefix
        base = base[0].lower() + base[1:]

    return prefix + base + suffix


# ── Public API ────────────────────────────────────────────────────────────────

def summarize(expression: str, gesture: str,
              action: str, sentiment: str,
              overall_conf: float) -> str:
    """
    Returns a one-sentence human-like description of the person's state.
    Language certainty scales with overall_conf:
      < 0.65  → vague   ("it seems like...")
      < 0.85  → hedged  ("it appears that...")
      ≥ 0.85  → direct  ("The person is...")
    """
    prompt = _build_prompt(expression, gesture, action, sentiment, overall_conf)
    result = _call_ollama(prompt)

    if result:
        print(f"[SUMMARY] Ollama ({_confidence_tier(overall_conf)}): {result}")
        return result

    result = _template_fallback(expression, gesture, action, sentiment, overall_conf)
    print(f"[SUMMARY] Template ({_confidence_tier(overall_conf)}): {result}")
    return result