"""
SOVA NLP Engine
Runs sentiment analysis on accumulated caption text using a local
DistilBERT model (~67 MB). No API key or internet connection required
after the first download.

The model is loaded lazily on the first real analyze() call — not at
import time — so importing this module never blocks app startup or
settings saves.
"""

from transformers import pipeline

_classifier = None  # loaded on first use, not at import time


def _get_classifier():
    global _classifier
    if _classifier is None:
        print("[NLP] Loading sentiment model...")
        _classifier = pipeline(
            task="sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
        print("[NLP] Model ready.")
    return _classifier


def analyze(audio: list[str]) -> tuple[str, float]:
    """
    Takes a list of raw caption strings from the current flush window.
    Returns (label, confidence) where label is 'positive', 'negative',
    or 'neutral'.
    """
    if not audio:
        print("[NLP] No audio detected.")
        return "Neutral", 1.0

    text = " ".join(audio).strip()
    if not text:
        print("[NLP] No audio detected.")
        return "Neutral", 1.0

    result = _get_classifier()(text)[0]

    label = result["label"].lower()   # → 'positive' or 'negative'
    conf  = round(result["score"], 3)

    if conf < 0.65:
        return "neutral", round(1.0 - conf, 3)

    return label, conf