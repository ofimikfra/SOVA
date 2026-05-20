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
import re

_NEUTRAL_RE = re.compile(
    r"\b(yeah|yes|yep|sure|okay|ok|alright|uh+|um+|hmm+|"
    r"i (think|mean|guess|see|know)|you know|let me|let's|"
    r"so|anyway|moving on|next|first|second|also|"
    r"the (meeting|call|slide|screen|document)|"
    r"looking at|going over|talking about|one moment|just a sec)\b",
    re.IGNORECASE
)

_FALSE_NEGATIVE_RE = re.compile(
    r"\b("
    r"not bad|not wrong|no problem|no issue|don't worry|"
    r"can't (find|see|hear|get|open|share|access)|"
    r"cannot (find|see|hear|get|open|share|access)|"
    r"issue|problem|bug|error|fix|concern|difficult|hard|wrong|"
    r"fail|broken|doesn't|isn't|aren't|won't|can't|don't|not"
    r")\b",
    re.IGNORECASE
)

_classifier = None

try:
    print("[NLP] Loading sentiment model...")
    _classifier = pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512,
    )
    print("[NLP] Model ready.")
except Exception as e:
    print(f"[NLP] Failed to load sentiment model: {e}")
    print("[NLP] Sentiment analysis will default to neutral.")


def analyze(audio: list[str]) -> tuple[str, float]:
    if _classifier is None:
        return "neutral", 0.90
    
    if not audio:
        print("[NLP] No audio detected.")
        return "Neutral", 1.0

    text = " ".join(audio).strip()
    if not text:
        print("[NLP] No audio detected.")
        return "Neutral", 1.0
    
    # If >30% of words match meeting/conversational filler, neutral 
    words = text.split()
    density = len(_NEUTRAL_RE.findall(text)) / max(len(words), 1)
    if density >= 0.30:
        return "neutral", round(min(0.70 + density * 0.20, 0.95), 3)

    result = _classifier(text)[0]
    label = result["label"].lower()
    conf  = round(result["score"], 3)

    # SST-2 false-negative override:
    # Work/meeting speech is full of negation words and task nouns that the
    # model was never trained to distinguish from genuinely negative sentiment.
    # If the negative label relies heavily on these words, downgrade to neutral.
    if label == "negative":
        words = text.split()
        neg_trigger_count = len(_FALSE_NEGATIVE_RE.findall(text))
        neg_density = neg_trigger_count / max(len(words), 1)
        
        # High density of work-context "negative-looking" words = likely neutral
        if neg_density >= 0.15 or conf < 0.88:
            return "neutral", round(1.0 - conf, 3)

    if conf < 0.82:
        return "neutral", round(1.0 - conf, 3)

    return label, conf