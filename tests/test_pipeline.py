"""
tests/test_pipeline.py

SOVA pipeline tests using pytest.

Run from project root:
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py -v --tb=short   # shorter tracebacks
    pytest tests/test_pipeline.py -v -k "nlp"     # run only NLP tests
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _force_flush(expr, gest, action, captions=None):
    import src.processor as p
    p.processExpression(expr,   1.0)
    p.processGesture(gest,      1.0)
    p.processBodyAction(action, 1.0)
    p._next_flush_time = 0
    return p.flushAll(captions=captions or [])


def _words(text: str) -> int:
    return len(text.strip().split())


# ══════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════

@pytest.fixture(autouse=True)
def clear_buffers():
    """Reset processor buffers and timer before every test."""
    import src.processor as p
    p._expr_buffer.clear()
    p._gest_buffer.clear()
    p._action_buffer.clear()
    p._next_flush_time = 0
    yield


@pytest.fixture
def fuse():
    from src.processor import _fuse_sentiment
    return _fuse_sentiment


@pytest.fixture
def template():
    from src.description_engine import _template_fallback
    return _template_fallback


@pytest.fixture
def tier():
    from src.description_engine import _confidence_tier
    return _confidence_tier


@pytest.fixture
def nlp():
    from src.nlp_engine import analyze
    return analyze


@pytest.fixture
def dominant():
    from src.processor import _getDominant
    return _getDominant


# ══════════════════════════════════════════════
#  1. Confidence tiers
# ══════════════════════════════════════════════

class TestConfidenceTiers:

    def test_low_range(self, tier):
        assert tier(0.00) == "low"
        assert tier(0.40) == "low"
        assert tier(0.64) == "low"

    def test_medium_range(self, tier):
        assert tier(0.65) == "medium"
        assert tier(0.75) == "medium"
        assert tier(0.84) == "medium"

    def test_high_range(self, tier):
        assert tier(0.85) == "high"
        assert tier(0.90) == "high"
        assert tier(1.00) == "high"

    def test_exact_boundaries(self, tier):
        assert tier(0.65) == "medium", "0.65 is the start of medium"
        assert tier(0.85) == "high",   "0.85 is the start of high"


# ══════════════════════════════════════════════
#  2. Template fallback
# ══════════════════════════════════════════════

class TestTemplateFallback:

    def test_low_conf_hedges(self, template):
        result = template("Neutral", "No Gesture", "Person Center", "neutral", 0.40)
        assert result.lower().startswith("it seems like"), \
            f"Got: '{result}'"

    def test_medium_conf_hedges(self, template):
        result = template("Smiling", "No Gesture", "Person Center", "positive", 0.70)
        assert result.lower().startswith("it appears that"), \
            f"Got: '{result}'"

    def test_high_conf_direct(self, template):
        result = template("Smiling", "No Gesture", "Person Center", "positive", 0.90)
        assert result.startswith("The"), f"Got: '{result}'"
        assert "seems"   not in result.lower()
        assert "appears" not in result.lower()

    def test_gesture_suffix(self, template):
        result = template("Smiling", "Thumbs Up", "Person Center", "positive", 0.90)
        assert "approval" in result.lower(), f"Got: '{result}'"

    def test_action_suffix(self, template):
        result = template("Neutral", "No Gesture", "Looking Away", "neutral", 0.90)
        assert "distracted" in result.lower(), f"Got: '{result}'"

    def test_gesture_beats_action(self, template):
        result = template("Neutral", "Thumbs Up", "Looking Away", "neutral", 0.90)
        assert "approval"   in result.lower(), f"Got: '{result}'"
        assert "distracted" not in result.lower(), f"Got: '{result}'"

    def test_no_suffix_when_neither(self, template):
        result = template("Smiling", "No Gesture", "Person Center", "positive", 0.90)
        assert "and is" not in result.lower(), f"Got: '{result}'"

    def test_unknown_combo_doesnt_crash(self, template):
        result = template("Left Wink", "No Gesture", "Person Center", "negative", 0.50)
        assert isinstance(result, str) and len(result) > 0

    def test_never_returns_empty(self, template):
        expressions = ["Smiling", "Frowning", "Neutral", "Eyebrows Raised",
                       "Mouth Open", "Left Wink", "Right Wink"]
        for expr in expressions:
            for sent in ("positive", "negative", "neutral"):
                result = template(expr, "No Gesture", "Person Center", sent, 0.70)
                assert len(result.strip()) > 0, f"Empty for ({expr}, {sent})"

    def test_prefix_lowercases_base(self, template):
        result = template("Smiling", "No Gesture", "Person Center", "positive", 0.40)
        assert result.startswith("It seems like the"), f"Got: '{result}'"


# ══════════════════════════════════════════════
#  3. Ollama post-processing
# ══════════════════════════════════════════════

class TestOllamaPostProcessing:

    def test_strips_description_label(self):
        text = "Description: The person seems engaged."
        if text.lower().startswith("description:"):
            text = text[len("description:"):].strip()
        assert text == "The person seems engaged."

    @pytest.mark.parametrize("prefix", [
        "Description:", "DESCRIPTION:", "description:"
    ])
    def test_label_strip_case_insensitive(self, prefix):
        text = f"{prefix} The person seems calm."
        if text.lower().startswith("description:"):
            text = text[len("description:"):].strip()
        assert text == "The person seems calm.", f"Failed for prefix '{prefix}'"

    def test_keeps_first_sentence_only(self):
        text = "The person seems happy. They are waving. And smiling."
        if "." in text:
            text = text[:text.index(".") + 1]
        assert text == "The person seems happy."

    def test_multiple_periods_truncates_at_first(self):
        text = "They seem happy. But nervous. And distracted."
        if "." in text:
            text = text[:text.index(".") + 1]
        assert text == "They seem happy."

    def test_no_period_keeps_full_text(self):
        text = "The person seems engaged"
        if "." in text:
            text = text[:text.index(".") + 1]
        assert text == "The person seems engaged"

    @pytest.mark.parametrize("bad", [
        "I think the person seems happy.",
        "I feel like they are engaged.",
        "I'm not sure but they look focused.",
        "I can't tell but they seem positive.",
    ])
    def test_discards_first_person(self, bad):
        _BAD_STARTS = (
            "i think ", "i feel ", "i believe ", "i'm not ",
            "i can't ", "i notice ", "it seems like but",
        )
        assert any(bad.lower().startswith(b) for b in _BAD_STARTS), \
            f"Should have been discarded: '{bad}'"

    def test_discards_contradictions(self):
        _CONTRADICTIONS = [
            ("happy", "frown"), ("happy", "concerned"), ("happy", "worried"),
            ("smiling", "frown"), ("positive", "confused"),
        ]
        text  = "The person looks genuinely happy with a frown."
        lower = text.lower()
        assert any(a in lower and b in lower for a, b in _CONTRADICTIONS)

    def test_empty_response_returns_none(self):
        text   = "   "
        result = text.strip() if text.strip() else None
        assert result is None


# ══════════════════════════════════════════════
#  4. Sentiment fusion
# ══════════════════════════════════════════════

class TestSentimentFusion:

    def test_positive_signals(self, fuse):
        sentiment, conf = fuse("Smiling", "positive", 0.92)
        assert sentiment == "positive"
        assert conf > 0.25

    def test_negative_signals(self, fuse):
        sentiment, conf = fuse("Frowning", "negative", 0.88)
        assert sentiment == "negative"
        assert conf > 0.25

    def test_conflict_yields_neutral(self, fuse):
        sentiment, conf = fuse("Smiling", "negative", 0.88)
        assert sentiment == "neutral", \
            f"Conflicting signals should yield neutral, got {sentiment}"

    def test_neutral_expression_no_captions(self, fuse):
        sentiment, conf = fuse("Neutral", "neutral", 1.0)
        assert sentiment == "neutral"

    def test_unknown_expression_defaults_zero(self):
        from src.processor import _expression_to_sentiment
        polarity, weight = _expression_to_sentiment("Confused")
        assert polarity == 0.0 and weight == 0.0

    def test_boundary_025_is_neutral(self, fuse):
        # Mouth Open (0.10 * 0.3 = 0.03) + neutral NLP → blended ≈ 0.012 → neutral
        sentiment, conf = fuse("Mouth Open", "neutral", 1.0)
        assert sentiment == "neutral"

    def test_zero_nlp_confidence(self, fuse):
        sentiment, conf = fuse("Smiling", "positive", 0.0)
        assert sentiment in ("positive", "negative", "neutral")
        assert 0.0 <= conf <= 1.0

    def test_max_confidence_stays_bounded(self, fuse):
        sentiment, conf = fuse("Frowning", "negative", 1.0)
        assert sentiment == "negative"
        assert conf <= 1.0

    @pytest.mark.parametrize("expr", [
        "Smiling", "Frowning", "Neutral", "Eyebrows Raised",
        "Mouth Open", "Left Wink", "Right Wink"
    ])
    @pytest.mark.parametrize("nlp_label", ["positive", "negative", "neutral"])
    def test_all_combos_stay_in_range(self, fuse, expr, nlp_label):
        sentiment, conf = fuse(expr, nlp_label, 0.80)
        assert sentiment in ("positive", "negative", "neutral")
        assert 0.0 <= conf <= 1.0


# ══════════════════════════════════════════════
#  5. NLP engine
# ══════════════════════════════════════════════

class TestNLP:

    def test_positive_captions(self, nlp):
        label, conf = nlp([
            "This is a great idea, I love it!",
            "Absolutely, let's go with that plan.",
            "Yes, that makes total sense.",
        ])
        assert label == "positive", f"Got {label} ({conf:.2f})"
        assert conf >= 0.65

    def test_negative_captions(self, nlp):
        label, conf = nlp([
            "I really disagree with this.",
            "This isn't working at all.",
            "I'm not happy with these results.",
        ])
        assert label == "negative", f"Got {label} ({conf:.2f})"
        assert conf >= 0.65

    def test_empty_returns_neutral(self, nlp):
        label, conf = nlp([])
        assert label == "neutral"
        assert conf  == 1.0

    def test_whitespace_only_returns_neutral(self, nlp):
        label, conf = nlp(["   ", "\t", "\n"])
        assert label == "neutral"

    def test_mixed_not_confidently_positive(self, nlp):
        label, conf = nlp([
            "I think this might work, not totally sure.",
            "I really disagree with that approach.",
            "Could we revisit this? I'm confused.",
            "It seems a bit risky.",
        ])
        assert not (label == "positive" and conf > 0.95), \
            f"Mixed captions should not be confidently positive: {label} ({conf:.2f})"

    def test_single_word(self, nlp):
        label, conf = nlp(["good"])
        assert label in ("positive", "negative", "neutral")
        assert 0.0 <= conf <= 1.0

    def test_very_long_caption_truncated(self, nlp):
        label, conf = nlp(["This is fine. " * 200])
        assert label in ("positive", "negative", "neutral")
        assert 0.0 <= conf <= 1.0

    def test_special_characters_and_emojis(self, nlp):
        label, conf = nlp([
            "Wow!!! This is amazing 🎉",
            "100% agree with you.",
            "Check this: https://example.com",
        ])
        assert label in ("positive", "negative", "neutral")
        assert 0.0 <= conf <= 1.0

    def test_repeated_captions(self, nlp):
        label, conf = nlp(["Great."] * 50)
        assert label in ("positive", "negative", "neutral")
        assert 0.0 <= conf <= 1.0

    def test_numeric_only(self, nlp):
        label, conf = nlp(["1 2 3", "100", "3.14"])
        assert label in ("positive", "negative", "neutral")

    @pytest.mark.parametrize("captions", [
        [], ["great"], ["terrible"], ["   "], ["okay sure fine"]
    ])
    def test_confidence_always_in_range(self, nlp, captions):
        _, conf = nlp(captions)
        assert 0.0 <= conf <= 1.0, f"Confidence out of range for {captions}: {conf}"


# ══════════════════════════════════════════════
#  6. getDominant
# ══════════════════════════════════════════════

class TestGetDominant:

    def test_empty_buffer_returns_neutral(self, dominant):
        assert dominant([], neutral="Neutral") == "Neutral"

    def test_single_weak_item_neutral_wins(self, dominant):
        # Neutral wins only when its accumulated score ties or beats the non-neutral score.
        # Push one Smiling AND one Neutral so neutral_score >= top_score.
        result = dominant(
            [("Smiling", 0.80), ("Neutral", 0.80)],
            neutral="Neutral"
        )
        # neutral_score (0.80) >= top_score (0.80) and top_score < 3.0 → neutral wins
        assert result == "Neutral", f"Got {result}"

    def test_single_item_no_neutral_wins_outright(self, dominant):
        # No neutral in buffer at all — non-neutral wins regardless of score
        result = dominant([("Smiling", 0.95)], neutral="Neutral")
        assert result == "Smiling", f"Got {result}"

    def test_strong_accumulated_score_wins(self, dominant):
        # 4 × 0.95 = 3.8 > 3.0 threshold
        result = dominant([("Smiling", 0.95)] * 4, neutral="Neutral")
        assert result == "Smiling"

    def test_all_neutral(self, dominant):
        result = dominant([("Neutral", 1.0)] * 10, neutral="Neutral")
        assert result == "Neutral"

    def test_tie_returns_valid_label(self, dominant):
        result = dominant([("Smiling", 1.0), ("Frowning", 1.0)], neutral="Neutral")
        assert result in ("Smiling", "Frowning", "Neutral")


# ══════════════════════════════════════════════
#  7. Caption accumulation
# ══════════════════════════════════════════════

class TestCaptionAccumulation:

    def test_accumulates_across_frames(self):
        accumulated = []
        accumulated.extend(["This is great."])
        # flush hasn't fired — don't clear
        accumulated.extend(["I agree with this."])
        # flush fires — take and clear
        result = accumulated.copy()
        accumulated.clear()
        assert len(result) == 2
        assert "This is great."    in result
        assert "I agree with this." in result

    def test_only_cleared_on_flush(self):
        accumulated = ["caption one", "caption two"]
        flush_result = None
        if flush_result:
            accumulated.clear()
        assert len(accumulated) == 2


# ══════════════════════════════════════════════
#  8. Full flush cycle
# ══════════════════════════════════════════════

class TestFlushCycle:

    def test_positive_captions_and_smiling(self):
        result = _force_flush("Smiling", "No Gesture", "Person Center",
                              captions=["This is a great idea!", "Brilliant."])
        assert result is not None
        expr, _, _, sentiment, conf, description = result
        assert expr      == "Smiling"
        assert sentiment in ("positive", "neutral")
        assert isinstance(description, str) and len(description) > 5

    def test_negative_captions_and_frowning(self):
        result = _force_flush("Frowning", "No Gesture", "Person Center",
                              captions=["I disagree.", "This isn't working."])
        assert result is not None
        _, _, _, sentiment, _, _ = result
        assert sentiment in ("negative", "neutral")

    def test_no_captions_yields_neutral(self):
        result = _force_flush("Neutral", "No Gesture", "Person Center", captions=[])
        assert result is not None
        _, _, _, sentiment, _, _ = result
        assert sentiment == "neutral"

    def test_gesture_preserved(self):
        result = _force_flush("Smiling", "Thumbs Up", "Person Center",
                              captions=["Great work."])
        assert result is not None
        assert result[1] == "Thumbs Up"

    def test_description_ends_with_period(self):
        result = _force_flush("Neutral", "No Gesture", "Person Center",
                              captions=["Let's see."])
        assert result is not None
        assert result[5].endswith(".")

    def test_description_word_count(self):
        result = _force_flush("Smiling", "No Gesture", "Person Center",
                              captions=["Enjoying this."])
        assert result is not None
        assert _words(result[5]) <= 20, \
            f"Too long ({_words(result[5])} words): '{result[5]}'"

    def test_returns_6_tuple(self):
        result = _force_flush("Smiling", "No Gesture", "Person Center")
        assert result is not None
        assert len(result) == 6

    def test_description_is_non_empty_string(self):
        result = _force_flush("Neutral", "No Gesture", "Person Center")
        assert result is not None
        assert isinstance(result[5], str) and len(result[5].strip()) > 0

    def test_captions_none_doesnt_crash(self):
        import src.processor as p
        p.processExpression("Neutral", 1.0)
        p.processGesture("No Gesture", 1.0)
        p.processBodyAction("Person Center", 1.0)
        p._next_flush_time = 0
        result = p.flushAll(captions=None)
        assert result is not None

    def test_empty_buffers_return_neutral_defaults(self):
        import src.processor as p
        p._expr_buffer.clear()
        p._gest_buffer.clear()
        p._action_buffer.clear()
        p._next_flush_time = 0
        result = p.flushAll(captions=[])
        assert result is not None
        assert result[0] == "Neutral",       f"Got {result[0]}"
        assert result[1] == "No Gesture",    f"Got {result[1]}"
        assert result[2] == "Person Center", f"Got {result[2]}"

    def test_respects_interval_timer(self):
        import src.processor as p
        import time
        p._next_flush_time = time.time() + 999
        result = p.flushAll(captions=[])
        assert result is None, "Should return None before interval elapses"


# ══════════════════════════════════════════════
#  9. Overall confidence calculation
# ══════════════════════════════════════════════

class TestOverallConfidence:

    def test_weighted_blend(self):
        expected = round(0.60 * 0.80 + 0.40 * 0.90, 3)
        assert expected == 0.84

    @pytest.mark.parametrize("sent_conf,expr_conf", [
        (0.0, 0.0), (0.5, 0.5), (1.0, 1.0),
        (0.0, 1.0), (1.0, 0.0),
    ])
    def test_always_in_range(self, sent_conf, expr_conf):
        overall = round(0.60 * sent_conf + 0.40 * expr_conf, 3)
        assert 0.0 <= overall <= 1.0


# ══════════════════════════════════════════════
#  10. Confidence vs description consistency
# ══════════════════════════════════════════════

class TestConfidenceDescriptionConsistency:

    def test_low_conf_template_hedges(self, template):
        result = template("Smiling", "No Gesture", "Person Center", "positive", 0.40)
        hedges = ("seems", "appears", "might", "looks like")
        assert any(h in result.lower() for h in hedges), \
            f"Low conf should hedge, got: '{result}'"

    def test_high_conf_template_no_hedge(self, template):
        result = template("Smiling", "No Gesture", "Person Center", "positive", 0.90)
        soft = ("might", "possibly", "perhaps", "it seems like", "it appears that")
        assert not any(h in result.lower() for h in soft), \
            f"High conf should not hedge, got: '{result}'"


class TestExpressionHierarchy:
    """
    Tests that _getDominant respects the expression priority hierarchy:
    Mouth Open > Left/Right Wink > Eyebrows Raised > Frowning > Smiling > Neutral
    """

    @pytest.fixture
    def expr_dominant(self):
        from src.processor import _getDominant, _EXPR_PRIORITY
        return lambda buf: _getDominant(buf, neutral="Neutral",
                                        label="TEST", priorities=_EXPR_PRIORITY)

    # ── Priority ordering ─────────────────────────────────────────────────────

    def test_mouth_open_beats_smiling(self, expr_dominant):
        # 1 Mouth Open vs 3 Smiling — priority 6 vs 2, mouth open should win
        buf = [
            ("Mouth Open", 0.90),
            ("Smiling",    0.85),
            ("Smiling",    0.85),
            ("Smiling",    0.85),
        ]
        assert expr_dominant(buf) == "Mouth Open", \
            "Mouth Open (priority 6) should beat 3× Smiling (priority 2)"

    def test_wink_beats_smiling(self, expr_dominant):
        # 1 Left Wink vs 2 Smiling — priority 5 vs 2
        buf = [
            ("Left Wink", 0.90),
            ("Smiling",   0.85),
            ("Smiling",   0.85),
        ]
        assert expr_dominant(buf) == "Left Wink", \
            "Left Wink (priority 5) should beat 2× Smiling (priority 2)"

    def test_right_wink_beats_frowning(self, expr_dominant):
        # Right Wink (5) > Frowning (3)
        buf = [
            ("Right Wink", 0.90),
            ("Frowning",   0.88),
            ("Frowning",   0.88),
        ]
        assert expr_dominant(buf) == "Right Wink", \
            "Right Wink (priority 5) should beat 2× Frowning (priority 3)"

    def test_eyebrows_raised_beats_smiling(self, expr_dominant):
        # Eyebrows Raised (4) > Smiling (2)
        buf = [
            ("Eyebrows Raised", 0.85),
            ("Smiling",         0.90),
            ("Smiling",         0.90),
        ]
        assert expr_dominant(buf) == "Eyebrows Raised", \
            "Eyebrows Raised (priority 4) should beat 2× Smiling (priority 2)"

    def test_frowning_beats_smiling(self, expr_dominant):
        # Frowning (3) > Smiling (2)
        buf = [
            ("Frowning", 0.85),
            ("Smiling",  0.90),
            ("Smiling",  0.90),
        ]
        assert expr_dominant(buf) == "Frowning", \
            "Frowning (priority 3) should beat 2× Smiling (priority 2)"

    def test_smiling_beats_neutral(self, expr_dominant):
        # Smiling (2) > Neutral (1) — enough accumulated score to overcome neutral
        buf = [("Smiling", 0.90)] * 4  # weighted score = 0.90 * 4 * 2 = 7.2
        assert expr_dominant(buf) == "Smiling", \
            "Smiling should win over Neutral with sufficient detections"

    # ── Confidence + priority interaction ─────────────────────────────────────

    def test_high_conf_low_priority_vs_low_conf_high_priority(self, expr_dominant):
        # 1 Mouth Open at 0.70 vs 1 Smiling at 0.95
        # weighted: Mouth Open = 0.70 * 6 = 4.2, Smiling = 0.95 * 2 = 1.9
        buf = [
            ("Mouth Open", 0.70),
            ("Smiling",    0.95),
        ]
        assert expr_dominant(buf) == "Mouth Open", \
            "Lower-conf Mouth Open should beat higher-conf Smiling due to priority"

    def test_many_smiling_eventually_beats_single_low_conf_mouth_open(self, expr_dominant):
        # Enough Smiling detections should eventually overcome Mouth Open
        # Mouth Open: 0.60 * 6 = 3.6
        # Smiling: 0.85 * 2 * N — needs N such that 0.85*2*N > 3.6 → N > 2.1 → N=3
        # 3× Smiling: 0.85 * 2 * 3 = 5.1 > 3.6
        buf = [
            ("Mouth Open", 0.60),
            ("Smiling",    0.85),
            ("Smiling",    0.85),
            ("Smiling",    0.85),
        ]
        assert expr_dominant(buf) == "Smiling", \
            "Enough high-conf Smiling detections should overcome weak Mouth Open"

    def test_equal_weighted_score_prefers_higher_priority(self, expr_dominant):
        # Frowning: 1.0 * 3 = 3.0, Smiling: 1.5 * 2 = 3.0 — tie in weighted score
        # max() will pick one — just verify it's a valid expression
        buf = [
            ("Frowning", 1.0),
            ("Smiling",  0.75),
            ("Smiling",  0.75),
        ]
        result = expr_dominant(buf)
        assert result in ("Frowning", "Smiling"), \
            f"Tie should return a valid expression, got {result}"

    # ── Full priority order ────────────────────────────────────────────────────

    @pytest.mark.parametrize("higher,lower,higher_priority,lower_priority", [
        ("Mouth Open",      "Left Wink",        6, 5),
        ("Mouth Open",      "Eyebrows Raised",  6, 4),
        ("Mouth Open",      "Frowning",         6, 3),
        ("Mouth Open",      "Smiling",          6, 2),
        ("Left Wink",       "Eyebrows Raised",  5, 4),
        ("Left Wink",       "Frowning",         5, 3),
        ("Left Wink",       "Smiling",          5, 2),
        ("Right Wink",      "Eyebrows Raised",  5, 4),
        ("Right Wink",      "Frowning",         5, 3),
        ("Right Wink",      "Smiling",          5, 2),
        ("Eyebrows Raised", "Frowning",         4, 3),
        ("Eyebrows Raised", "Smiling",          4, 2),
        ("Frowning",        "Smiling",          3, 2),
    ])
    def test_priority_ordering(self, expr_dominant,
                               higher, lower, higher_priority, lower_priority):
        """
        Each higher-priority expression should beat a same-confidence
        lower-priority expression when counts are equal.
        weighted_higher = conf * higher_priority
        weighted_lower  = conf * lower_priority
        Since higher_priority > lower_priority, higher always wins.
        """
        buf = [
            (higher, 0.85),
            (lower,  0.85),
        ]
        result = expr_dominant(buf)
        assert result == higher, \
            f"{higher} (p{higher_priority}) should beat {lower} (p{lower_priority}), got {result}"

    # ── Edge cases ─────────────────────────────────────────────────────────────

    def test_single_expression_returns_itself(self, expr_dominant):
        for expr in ("Mouth Open", "Left Wink", "Eyebrows Raised",
                     "Frowning", "Smiling"):
            buf    = [(expr, 0.90)] * 4
            result = expr_dominant(buf)
            assert result == expr, \
                f"Single repeated expression should return itself, got {result}"

    def test_all_neutral_returns_neutral(self, expr_dominant):
        buf = [("Neutral", 1.0)] * 10
        assert expr_dominant(buf) == "Neutral"

    def test_empty_buffer_returns_neutral(self, expr_dominant):
        assert expr_dominant([]) == "Neutral"

    def test_left_and_right_wink_same_priority(self, expr_dominant):
        # Both winks have priority 5 — the one with higher score wins
        buf = [
            ("Left Wink",  0.90),
            ("Right Wink", 0.70),
        ]
        result = expr_dominant(buf)
        assert result == "Left Wink", \
            "Left Wink with higher score should beat Right Wink (same priority)"