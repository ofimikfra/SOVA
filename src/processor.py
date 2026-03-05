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
# Returns a (polarity_score, weight) tuple.
# polarity_score: -1.0 (negative) to +1.0 (positive)
# weight: how strongly this expression signals sentiment (0–1)

_EXPR_POLARITY = {
    "Smiling":          ( 0.85, 0.9),
    "Eyebrows Raised":  ( 0.40, 0.5),  # surprise — weakly positive
    "Left Wink":        ( 0.60, 0.6),
    "Right Wink":       ( 0.60, 0.6),
    "Mouth Open":       ( 0.10, 0.3),  # ambiguous — near neutral
    "Frowning":         (-0.85, 0.9),
    "Neutral":          ( 0.00, 0.0),  # no signal
}

# ── Expression priority hierarchy ──────────────────────────────────────────────
# Higher number = higher priority. Applied as a multiplier to accumulated score
# so a high-priority expression can win even with fewer detections.
# Only used for expression buffers — gestures/actions are score-only.

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
    """Returns (polarity_score, signal_weight) for a given expression label."""
    return _EXPR_POLARITY.get(expression, (0.0, 0.0))


def _fuse_sentiment(expression: str,
                    nlp_label: str,
                    nlp_conf: float) -> tuple[str, float]:
    """
    Blends NLP sentiment (60%) with facial expression sentiment (40%)
    into a single unified label + confidence.

    NLP label is 'positive', 'negative', or 'neutral'.
    Returns (unified_label, unified_confidence).
    """

    # Convert NLP output to a -1 → +1 score
    if nlp_label == "positive":
        nlp_score = nlp_conf          #  0.65 → +1.0
    elif nlp_label == "negative":
        nlp_score = -nlp_conf         # -0.65 → -1.0
    else:
        nlp_score = 0.0               # neutral

    # Convert expression to a -1 → +1 score
    expr_polarity, expr_weight = _expression_to_sentiment(expression)
    # Scale by how strongly the expression signals sentiment
    expr_score = expr_polarity * expr_weight

    # Weighted blend: 60% NLP, 40% expression
    NLP_WEIGHT  = 0.60
    EXPR_WEIGHT = 0.40
    blended = (nlp_score * NLP_WEIGHT) + (expr_score * EXPR_WEIGHT)

    # Map blended score → label
    # Thresholds: |blended| > 0.25 = clear signal, else neutral
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

    # ── Capture conf before clearing buffers ──────────────────────────────
    expr_confs    = [c for _, c in _expr_buffer] if _expr_buffer else [1.0]
    avg_expr_conf = sum(expr_confs) / len(expr_confs)

    expression = _getDominant(
        _expr_buffer,
        neutral="Neutral",
        label="EXPRESSION",
        priorities=_EXPR_PRIORITY,
    )
    gesture = _getDominant(_gest_buffer,  neutral="No Gesture",    label="GESTURE")
    action  = _getDominant(_action_buffer, neutral="Person Center", label="ACTION")

    _expr_buffer.clear()
    _gest_buffer.clear()
    _action_buffer.clear()

    _next_flush_time = time.time() + INTERVAL

    nlp_label, nlp_conf  = nlp_analyze(captions or [])
    sentiment, sent_conf = _fuse_sentiment(expression, nlp_label, nlp_conf)

    overall_conf = round(0.60 * sent_conf + 0.40 * avg_expr_conf, 3)

    # In flushAll():
    had_captions = bool(captions)

    description = summarize(
        expression, gesture, action,
        nlp_label   = nlp_label  if had_captions else None,
        nlp_conf    = nlp_conf   if had_captions else None,
        overall_conf= overall_conf,
    )

    print(f"\n{'='*50}")
    print(f"[FLUSH] Results after {INTERVAL:.0f}s interval:")
    print(f"  Expression : {expression}")
    print(f"  Gesture    : {gesture}")
    print(f"  Action     : {action}")
    print(f"  NLP raw    : {nlp_label} ({nlp_conf:.2f})")
    print(f"  Sentiment  : {sentiment} ({sent_conf:.2f})  ← fused")
    print(f"  Confidence : {overall_conf:.2f} ({_confidence_tier(overall_conf)})")
    print(f"  Description: {description}")
    print(f"{'='*50}\n")

    return expression, gesture, action, sentiment, sent_conf, description


def set_interval(seconds: float):
    global INTERVAL, _next_flush_time
    INTERVAL = seconds
    print(f"[PROCESSOR] Flush interval set to {seconds}s")


# ── Internal ───────────────────────────────────────────────────────────────────

def _getDominant(buffer: list, neutral: str,
                 label: str = "",
                 priorities: dict | None = None) -> str:
    if not buffer:
        print(f"  [{label}] Buffer empty → {neutral}")
        return neutral

    scores = defaultdict(float)
    counts = defaultdict(int)

    for lbl, conf in buffer:
        scores[lbl] += conf
        counts[lbl] += 1

    # Apply priority multipliers if provided
    weighted = {
        lbl: score * priorities.get(lbl, 1)
        for lbl, score in scores.items()
    } if priorities else scores

    non_neutral = {k: v for k, v in weighted.items() if k != neutral}

    if non_neutral:
        top           = max(non_neutral, key=non_neutral.get)
        top_score     = non_neutral[top]
        neutral_score = weighted.get(neutral, 0.0)

        if neutral_score >= top_score:
            if top_score >= 3.0:
                return top
            return neutral

    return max(weighted, key=weighted.get)