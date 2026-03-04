"""
SOVA NLP Engine
Runs sentiment analysis on accumulated caption text using a local
DistilBERT model (~67 MB). No API key or internet connection required
after the first download.
"""

from transformers import pipeline

# Loads once at import time. First run downloads the model to ~/.cache/huggingface
print("[NLP] Loading sentiment model...")
_classifier = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512,
)
print("[NLP] Model ready.")


def analyze(audio: list[str]) -> tuple[str, float]:
    """
    Takes a list of raw caption strings from the current flush window.
    Returns (label, confidence) where label is 'positive', 'negative',
    or 'neutral'.
    """
    if not audio:
        print("[NLP] No audio detected.")
        return "Neutral", 1.0

    # Join into one block — DistilBERT handles up to 512 tokens
    text = " ".join(audio).strip()
    if not text:
        print("[NLP] No audio detected.")
        return "Neutral", 1.0

    result = _classifier(text)[0]

    # HuggingFace SST-2 returns 'POSITIVE' or 'NEGATIVE'
    label = result["label"].lower()   # → 'positive' or 'negative'
    conf  = round(result["score"], 3)

    # Anything below 0.65 confidence we call neutral to avoid noise
    if conf < 0.65:
        return "neutral", round(1.0 - conf, 3)

    return label, conf