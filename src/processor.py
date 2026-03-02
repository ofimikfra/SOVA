import time
from collections import defaultdict

INTERVAL = 5.0  # seconds

_next_flush_time = time.time() + INTERVAL

_expr_buffer   = []
_gest_buffer   = []
_action_buffer = []


# ------------------------------------ API ----------------------------------- #

def processExpression(expression: str, confidence: float = 1.0):
    _expr_buffer.append((expression, confidence))


def processGesture(gesture: str, confidence: float = 1.0):
    _gest_buffer.append((gesture, confidence))


def processBodyAction(action: str, confidence: float = 1.0):
    _action_buffer.append((action, confidence))

# ----------------------------- helper functions ----------------------------- #
    
# returns dominant expression, gesture, action after interval -> flushes channels

def flushAll() -> tuple | None:
    """
    Call once per frame. Returns (expression, gesture, action) when the
    30-second interval has elapsed, otherwise returns None.
    All three channels always flush together.
    """
    global _next_flush_time

    if time.time() < _next_flush_time:
        return None

    expression = _getDominant(_expr_buffer,   neutral="Neutral",       label="EXPRESSION")
    gesture    = _getDominant(_gest_buffer,    neutral="No Gesture",    label="GESTURE")
    action     = _getDominant(_action_buffer,  neutral="Person Center", label="ACTION")

    _expr_buffer.clear()
    _gest_buffer.clear()
    _action_buffer.clear()

    _next_flush_time = time.time() + INTERVAL

    # debugging
    print(f"\n{'='*50}")
    print(f"[FLUSH] Results after {INTERVAL:.0f}s interval:")
    print(f"  Expression : {expression}")
    print(f"  Gesture    : {gesture}")
    print(f"  Action     : {action}")
    print(f"{'='*50}\n")

    return expression, gesture, action


# calculate confidence score of each detected expression + no. detections -> analyze weightage -> output dominant 

def _getDominant(buffer: list, neutral: str, label: str = "") -> str:
    if not buffer:
        print(f"  [{label}] Buffer empty → {neutral}")
        return neutral

    scores = defaultdict(float)
    counts = defaultdict(int)

    for lbl, conf in buffer:
        scores[lbl] += conf
        counts[lbl] += 1

    non_neutral = {k: v for k, v in scores.items() if k != neutral}

    if non_neutral:
        top            = max(non_neutral, key=non_neutral.get)
        top_score      = non_neutral[top]
        neutral_score  = scores.get(neutral, 0.0)

        if neutral_score >= top_score:
            if top_score >= 3.0:
                return top
            return neutral

    result = max(scores, key=scores.get)
    return result

'''
TODO: create hierarchy of expressions 
      left wink = right wink < mouth open < eyebrows raised < smiling < neutral

TODO: detect multiple faces w/ gestures & body actions linked to faces 
'''