"""
tests/test_pipeline.py
Tests the SOVA processing pipeline end-to-end with synthetic data.
Run from project root: python -m pytest tests/ -v
  or directly:         python tests/test_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.processor import (
    processExpression, processGesture, processBodyAction,
    flushAll, set_interval, _fuse_sentiment
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _force_flush(expr, gest, action, captions=None):
    """Push one set of signals and force an immediate flush."""
    import src.processor as p
    processExpression(expr,   1.0)
    processGesture(gest,      1.0)
    processBodyAction(action, 1.0)
    # Force the timer so flushAll fires immediately
    p._next_flush_time = 0
    return flushAll(captions=captions or [])


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_basic_flush():
    result = _force_flush("Smiling", "No Gesture", "Person Center",
                          captions=["This is great, I love it!"])
    assert result is not None, "flushAll returned None"
    expr, gest, act, sentiment, conf, description = result
    assert expr == "Smiling"
    assert isinstance(description, str) and len(description) > 0
    print(f"  ✓ basic flush: '{description}'")


def test_sentiment_fusion_positive():
    sentiment, conf = _fuse_sentiment("Smiling", "positive", 0.92)
    assert sentiment == "positive", f"Expected positive, got {sentiment}"
    assert conf > 0.25
    print(f"  ✓ positive fusion: {sentiment} ({conf:.2f})")


def test_sentiment_fusion_conflict():
    # Smiling face but negative caption — should land near neutral
    sentiment, conf = _fuse_sentiment("Smiling", "negative", 0.80)
    # blended = (-0.80 * 0.6) + (0.85*0.9 * 0.4) = -0.48 + 0.306 = -0.174 → neutral
    assert sentiment == "neutral", f"Expected neutral on conflict, got {sentiment}"
    print(f"  ✓ conflict → neutral: {sentiment} ({conf:.2f})")


def test_sentiment_fusion_negative():
    sentiment, conf = _fuse_sentiment("Frowning", "negative", 0.88)
    assert sentiment == "negative", f"Expected negative, got {sentiment}"
    print(f"  ✓ negative fusion: {sentiment} ({conf:.2f})")


def test_no_captions_neutral():
    result = _force_flush("Neutral", "No Gesture", "Person Center", captions=[])
    assert result is not None
    _, _, _, sentiment, _, description = result
    print(f"  ✓ no captions → sentiment={sentiment}, desc='{description}'")


def test_description_not_empty():
    result = _force_flush("Frowning", "Thumbs Down", "Looking Away",
                          captions=["I really disagree with this approach."])
    assert result is not None
    description = result[5]
    assert isinstance(description, str)
    assert len(description) > 10
    print(f"  ✓ description: '{description}'")


def test_confidence_tiers():
    from src.description_engine import _confidence_tier
    assert _confidence_tier(0.50) == "low"
    assert _confidence_tier(0.70) == "medium"
    assert _confidence_tier(0.90) == "high"
    print("  ✓ confidence tiers correct")


def test_template_fallback():
    """Verifies template fallback produces hedged language at low confidence."""
    from src.description_engine import _template_fallback
    result = _template_fallback("Neutral", "No Gesture", "Person Center",
                                "neutral", 0.40)
    assert result.startswith("It seems like"), f"Expected hedge, got: '{result}'"
    print(f"  ✓ low conf template: '{result}'")

    result_high = _template_fallback("Smiling", "No Gesture", "Person Center",
                                     "positive", 0.90)
    assert result_high.startswith("The"), f"Expected direct, got: '{result_high}'"
    print(f"  ✓ high conf template: '{result_high}'")


if __name__ == "__main__":
    tests = [
        test_confidence_tiers,
        test_template_fallback,
        test_sentiment_fusion_positive,
        test_sentiment_fusion_conflict,
        test_sentiment_fusion_negative,
        test_no_captions_neutral,
        test_basic_flush,
        test_description_not_empty,
    ]
    print("\n── SOVA Pipeline Tests ──────────────────────────\n")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed\n")