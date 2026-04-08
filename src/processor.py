import time
from collections import defaultdict
from models.nlp_engine import analyze as nlp_analyze
from src.description_engine import summarize, _confidence_tier

INTERVAL = 30.0  # seconds

_next_flush_time = time.time() + INTERVAL

_expr_buffer   = []
_gest_buffer   = []
_action_buffer = []

# ── Expression → sentiment polarity map ───────────────────────────────────────
_EXPR_POLARITY = {
    "Smiling":          ( 0.85, 0.9),
    "Eyebrows Raised":  ( 0.40, 0.5),
    "Left Wink":        ( 0.60, 0.6),
    "Right Wink":       ( 0.60, 0.6),
    "Mouth Open":       ( 0.10, 0.3),
    "Frowning":         (-0.85, 0.9),
    "Neutral":          ( 0.00, 0.0),
}

_EXPR_PRIORITY = {
    "Mouth Open":       5,
    "Left Wink":        4,
    "Right Wink":       4,
    "Eyebrows Raised":  3,
    "Frowning":         2,
    "Smiling":          2,
    "Neutral":          1,
}

def _expression_to_sentiment(expression: str) -> tuple[float, float]:
    return _EXPR_POLARITY.get(expression, (0.0, 0.0))

def _fuse_sentiment(expression: str, nlp_label: str, nlp_conf: float) -> tuple[str, float]:
    if nlp_label == "positive":
        nlp_score = nlp_conf
    elif nlp_label == "negative":
        nlp_score = -nlp_conf
    else:
        nlp_score = 0.0

    expr_polarity, expr_weight = _expression_to_sentiment(expression)
    expr_score = expr_polarity * expr_weight

    NLP_WEIGHT  = 0.60
    EXPR_WEIGHT = 0.40
    blended = (nlp_score * NLP_WEIGHT) + (expr_score * EXPR_WEIGHT)

    if blended > 0.25:
        label = "positive"
        conf  = round(min(blended, 1.0), 3)
    elif blended < -0.25:
        label = "negative"
        conf  = round(min(abs(blended), 1.0), 3)
    else:
        label = "neutral"
        conf  = round(1.0 - abs(blended), 3)

    return label, conf

# ── Internal Logic ─────────────────────────────────────────────────────

def _getDominant(buffer: list, neutral: str,
                 lbl: str = "",
                 priorities: dict | None = None,
                 neutral_threshold: float = 0.60) -> str: # Added threshold param
    if not buffer:
        return neutral

    scores = defaultdict(float)
    counts = defaultdict(int)
    total_items = len(buffer)

    for lbl, conf in buffer:
        scores[lbl] += conf
        counts[lbl] += 1

    # Use the specific threshold passed for this category
    neutral_count = counts.get(neutral, 0)
    if (neutral_count / total_items) >= neutral_threshold:
        return neutral

    # Otherwise, find the strongest non-neutral signal
    weighted = {
        lbl: score * priorities.get(lbl, 1)
        for lbl, score in scores.items()
    } if priorities else scores

    non_neutral = {k: v for k, v in weighted.items() if k != neutral}
    if not non_neutral:
        return neutral
    
    print(max(non_neutral))
    print(neutral_count)
    print(total_items)
    print(neutral_threshold)
    print(neutral_count/total_items)
    print((neutral_count / total_items) >= neutral_threshold)
    print()

    return max(non_neutral, key=non_neutral.get)

# ── Public API ─────────────────────────────────────────────────────────────────

def processExpression(expression: str, confidence: float = 1.0):
    _expr_buffer.append((expression, confidence))

def processGesture(gesture: str, confidence: float = 1.0):
    _gest_buffer.append((gesture, confidence))

def processBodyAction(action: str, confidence: float = 1.0):
    _action_buffer.append((action, confidence))

def flushAll(captions: list[str] | None = None) -> tuple | None:
    global _next_flush_time

    if time.time() < _next_flush_time:
        return None

    expr_confs    = [c for _, c in _expr_buffer] if _expr_buffer else [1.0]
    avg_expr_conf = sum(expr_confs) / len(expr_confs)

    expression = _getDominant(
        _expr_buffer,  
        neutral="Neutral",    
        lbl="EXPRESSION",
        neutral_threshold=0.60 # default threshold
    )

    gesture = _getDominant(
        _gest_buffer,  
        neutral="No Gesture",    
        lbl="GESTURE",
        neutral_threshold=0.95  # more sensitive threshold
    )

    action = _getDominant(
        _action_buffer, 
        neutral="No Person In Frame", 
        lbl="ACTION",
        neutral_threshold=0.70 # stricter threshold
    )

    _expr_buffer.clear()
    _gest_buffer.clear()
    _action_buffer.clear()

    _next_flush_time = time.time() + INTERVAL

    nlp_label, nlp_conf  = nlp_analyze(captions or [])
    sentiment, sent_conf = _fuse_sentiment(expression, nlp_label, nlp_conf)
    overall_conf = round(0.60 * sent_conf + 0.40 * avg_expr_conf, 3)

    description = summarize(
        expression, gesture, action,
        nlp_label   = nlp_label  if bool(captions) else None,
        nlp_conf    = nlp_conf   if bool(captions) else None,
        overall_conf= overall_conf,
    )

    print(f"\n{'='*50}")
    print(f"[FLUSH] Results after {INTERVAL:.0f}s interval:")
    print(f"  Expression : {expression}")
    print(f"  Gesture    : {gesture}")
    print(f"  Action     : {action}")
    print(f"  Sentiment  : {sentiment} ({sent_conf:.2f})")
    print(f"  Confidence : {overall_conf:.2f}")
    print(f"  Description: {description}")
    print(f"{'='*50}\n")

    return expression, gesture, action, sentiment, sent_conf, description

def set_interval(seconds: float):
    global INTERVAL, _next_flush_time
    INTERVAL = seconds
    _next_flush_time = time.time() + INTERVAL

def reset_timer():
    global _next_flush_time
    _next_flush_time = time.time() + INTERVAL