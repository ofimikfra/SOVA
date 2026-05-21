import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.description_engine import summarize
import numpy as np

from models.ollama import ensure_ollama
from src import description_engine as _desc
ready = ensure_ollama("llama3.2:3b")
_desc._ollama_ready = ready
if not ready:
    print("[SOVA] ⚠️  Ollama unavailable — descriptions will use templates.") 


EXPRESSIONS = [
    "Smiling",
    "Frowning",
    "Mouth Open",
    "Eyebrows Raised",
    "Left Wink",
    "Right Wink",
    "Neutral"
]

GESTURES = [
    "Thumbs Up",
    "Thumbs Down",
    "Waving",
    "Pointing",
    "Peace Sign",
    "OK Sign",
    "No Gesture"
]

ACTIONS = [
    "Looking Away",
    "No Person In Frame",
    "Person Present"
]

NLP_LABEL = ["positive", "negative", None]

CONF = [
    0.3, 0.65, 0.8, 0.95
]

rng = np.random.default_rng()

nlp_label = NLP_LABEL[rng.integers(low=0, high=len(NLP_LABEL))]
nlp_conf = CONF[rng.integers(low=0, high=len(CONF))] if nlp_label is not None else None

expression = EXPRESSIONS[rng.integers(low=0, high=len(EXPRESSIONS))]
gesture = GESTURES[rng.integers(low=0, high=len(GESTURES))]
action = ACTIONS[rng.integers(low=0, high=len(ACTIONS))]
overall_conf = CONF[rng.integers(low=0, high=len(CONF))]

print(f"\nexpression: {expression}")
print(f"gesture: {gesture}")
print(f"action: {action}")
print(f"nlp_label: {nlp_label}")
print(f"nlp_conf: {nlp_conf}")
print(f"overall_conf: {overall_conf}")

summarize(expression, gesture, action,
        nlp_label, nlp_conf,
            overall_conf)
