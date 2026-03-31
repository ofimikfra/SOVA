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
    """
    Push one set of signals and force an immediate flush.

    captions=None  → no captions this window (had_captions=False)
                     summarize() receives nlp_label=None
    captions=[]    → same as None — no speech signal passed to Ollama
    captions=[...] → captions present (had_captions=True)
                     NLP runs and nlp_label is passed to summarize()
    """
    import src.processor as p
    p.processExpression(expr,   1.0)
    p.processGesture(gest,      1.0)
    p.processBodyAction(action, 1.0)
    p._next_flush_time = 0
    return p.flushAll(captions=captions)


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
    from models.nlp_engine import analyze
    return analyze


@pytest.fixture
def dominant():
    from src.processor import _getDominant
    return _getDominant


@pytest.fixture
def build_prompt():
    from src.description_engine import _build_prompt
    return _build_prompt


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
#  2. Template fallback — hedging by tier
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
#  3. Template fallback — no captions path
#  When nlp_label=None, fallback derives sentiment
#  from _EXPR_POLARITY directly instead of fusion.
# ══════════════════════════════════════════════

class TestTemplateFallbackNoCaptions:

    def test_smiling_no_captions_uses_positive_template(self, template):
        """Smiling has positive polarity (0.85) — should map to positive template."""
        result = template("Smiling", "No Gesture", "Person Center", "positive", 0.90)
        assert isinstance(result, str) and len(result) > 0
        # High conf + positive → direct statement
        assert result.startswith("The"), f"Got: '{result}'"

    def test_frowning_no_captions_uses_negative_template(self, template):
        """Frowning has negative polarity (-0.85) — should map to negative template."""
        result = template("Frowning", "No Gesture", "Person Center", "negative", 0.90)
        assert isinstance(result, str) and len(result) > 0
        assert "concerned" in result.lower() or "displeased" in result.lower(), \
            f"Got: '{result}'"

    def test_neutral_no_captions_uses_neutral_template(self, template):
        """Neutral expression has zero polarity — should map to neutral template."""
        result = template("Neutral", "No Gesture", "Person Center", "neutral", 0.90)
        assert isinstance(result, str) and len(result) > 0
        assert "focused" in result.lower() or "attentive" in result.lower(), \
            f"Got: '{result}'"

    def test_expression_polarity_mapping(self):
        """Verify _EXPR_POLARITY maps expressions to the right sentiment bucket."""
        from src.processor import _EXPR_POLARITY

        positive_exprs = ["Smiling", "Eyebrows Raised", "Left Wink", "Right Wink"]
        negative_exprs = ["Frowning"]
        neutral_exprs  = ["Neutral", "Mouth Open"]

        for expr in positive_exprs:
            polarity, _ = _EXPR_POLARITY.get(expr, (0.0, 0.0))
            assert polarity > 0, \
                f"{expr} should have positive polarity, got {polarity}"

        for expr in negative_exprs:
            polarity, _ = _EXPR_POLARITY.get(expr, (0.0, 0.0))
            assert polarity < 0, \
                f"{expr} should have negative polarity, got {polarity}"

        for expr in neutral_exprs:
            polarity, _ = _EXPR_POLARITY.get(expr, (0.0, 0.0))
            assert abs(polarity) <= 0.25, \
                f"{expr} should be near-neutral polarity, got {polarity}"


# ══════════════════════════════════════════════
#  4. Prompt building
#  Tests _build_prompt() directly to verify that
#  speech lines and conflict notes are included
#  or omitted correctly based on inputs.
# ══════════════════════════════════════════════

class TestPromptBuilding:

    def test_no_captions_omits_speech_line(self, build_prompt):
        """When nlp_label=None (no captions), prompt should contain no speech sentiment line."""
        prompt = build_prompt(
            "Smiling", "No Gesture", "Person Center",
            nlp_label=None, nlp_conf=None, overall_conf=0.80
        )
        assert "Speech sentiment" not in prompt, \
            f"Speech line should be absent when nlp_label=None, got:\n{prompt}"

    def test_captions_present_includes_speech_line(self, build_prompt):
        """When nlp_label is set, prompt should include the speech sentiment line."""
        prompt = build_prompt(
            "Neutral", "No Gesture", "Person Center",
            nlp_label="positive", nlp_conf=0.88, overall_conf=0.80
        )
        assert "Speech sentiment" in prompt, \
            f"Speech line should be present when nlp_label is set, got:\n{prompt}"
        assert "positive" in prompt.lower(), \
            f"Speech label should appear in prompt, got:\n{prompt}"

    def test_conflict_smiling_negative_adds_note(self, build_prompt):
        """Smiling + negative speech should trigger conflict note in prompt."""
        prompt = build_prompt(
            "Smiling", "No Gesture", "Person Center",
            nlp_label="negative", nlp_conf=0.85, overall_conf=0.60
        )
        assert "conflict" in prompt.lower(), \
            f"Conflict note should appear for smiling + negative speech, got:\n{prompt}"
        assert "sarcasm" in prompt.lower() or "mixed" in prompt.lower(), \
            f"Conflict note should mention sarcasm or mixed feelings, got:\n{prompt}"

    def test_conflict_frowning_positive_adds_note(self, build_prompt):
        """Frowning + positive speech should trigger conflict note in prompt."""
        prompt = build_prompt(
            "Frowning", "No Gesture", "Person Center",
            nlp_label="positive", nlp_conf=0.85, overall_conf=0.60
        )
        assert "conflict" in prompt.lower(), \
            f"Conflict note should appear for frowning + positive speech, got:\n{prompt}"

    def test_no_conflict_when_signals_align(self, build_prompt):
        """Smiling + positive speech — no conflict note should appear."""
        prompt = build_prompt(
            "Smiling", "No Gesture", "Person Center",
            nlp_label="positive", nlp_conf=0.88, overall_conf=0.85
        )
        assert "conflict" not in prompt.lower(), \
            f"No conflict note expected for aligned signals, got:\n{prompt}"

    def test_no_conflict_note_when_no_captions(self, build_prompt):
        """With no captions, there can be no conflict — note must not appear."""
        prompt = build_prompt(
            "Smiling", "No Gesture", "Person Center",
            nlp_label=None, nlp_conf=None, overall_conf=0.80
        )
        assert "conflict" not in prompt.lower(), \
            f"No conflict note expected when nlp_label=None, got:\n{prompt}"

    def test_gesture_line_included_when_present(self, build_prompt):
        """Non-default gesture should appear in the prompt."""
        prompt = build_prompt(
            "Smiling", "Thumbs Up", "Person Center",
            nlp_label=None, nlp_conf=None, overall_conf=0.80
        )
        assert "Thumbs Up" in prompt, \
            f"Gesture should appear in prompt, got:\n{prompt}"

    def test_gesture_line_omitted_when_default(self, build_prompt):
        """'No Gesture' should not appear as a signal line in the prompt."""
        prompt = build_prompt(
            "Smiling", "No Gesture", "Person Center",
            nlp_label=None, nlp_conf=None, overall_conf=0.80
        )
        assert "No Gesture" not in prompt, \
            f"Default gesture should be omitted from prompt, got:\n{prompt}"

    def test_action_line_omitted_when_default(self, build_prompt):
        """'Person Center' should not appear as a signal line in the prompt."""
        prompt = build_prompt(
            "Smiling", "No Gesture", "Person Center",
            nlp_label=None, nlp_conf=None, overall_conf=0.80
        )
        assert "Person Center" not in prompt, \
            f"Default action should be omitted from prompt, got:\n{prompt}"

    def test_action_line_included_when_non_default(self, build_prompt):
        """Non-default action should appear in the prompt."""
        prompt = build_prompt(
            "Neutral", "No Gesture", "Looking Away",
            nlp_label=None, nlp_conf=None, overall_conf=0.80
        )
        assert "Looking Away" in prompt, \
            f"Non-default action should appear in prompt, got:\n{prompt}"

    def test_prompt_always_contains_expression(self, build_prompt):
        """Expression should always appear in the prompt."""
        for expr in ("Smiling", "Frowning", "Neutral", "Eyebrows Raised"):
            prompt = build_prompt(
                expr, "No Gesture", "Person Center",
                nlp_label=None, nlp_conf=None, overall_conf=0.80
            )
            assert expr in prompt, \
                f"Expression '{expr}' should always appear in prompt"

    def test_prompt_contains_confidence(self, build_prompt):
        """Confidence value should appear in the prompt."""
        prompt = build_prompt(
            "Neutral", "No Gesture", "Person Center",
            nlp_label=None, nlp_conf=None, overall_conf=0.75
        )
        assert "75%" in prompt or "Confidence" in prompt, \
            f"Confidence should appear in prompt, got:\n{prompt}"

    @pytest.mark.parametrize("nlp_label,nlp_conf", [
        ("positive", 0.90),
        ("negative", 0.80),
        ("neutral",  0.70),
    ])
    def test_speech_confidence_shown_as_percentage(self, build_prompt, nlp_label, nlp_conf):
        """Speech confidence should be shown as a percentage in the prompt."""
        prompt = build_prompt(
            "Neutral", "No Gesture", "Person Center",
            nlp_label=nlp_label, nlp_conf=nlp_conf, overall_conf=0.75
        )
        expected_pct = f"{int(nlp_conf * 100)}%"
        assert expected_pct in prompt, \
            f"Expected '{expected_pct}' in prompt, got:\n{prompt}"


# ══════════════════════════════════════════════
#  5. Ollama post-processing
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
#  6. Sentiment fusion
#  Fusion is now used for dashboard UI only —
#  not passed to Ollama. Tests remain to ensure
#  the dashboard sentiment label stays correct.
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
        # Fusion still collapses conflicts to neutral for the dashboard colour
        sentiment, conf = fuse("Smiling", "negative", 0.88)
        assert sentiment == "neutral", \
            f"Conflicting signals should yield neutral for dashboard, got {sentiment}"

    def test_neutral_expression_no_captions(self, fuse):
        sentiment, conf = fuse("Neutral", "neutral", 1.0)
        assert sentiment == "neutral"

    def test_unknown_expression_defaults_zero(self):
        from src.processor import _expression_to_sentiment
        polarity, weight = _expression_to_sentiment("Confused")
        assert polarity == 0.0 and weight == 0.0

    def test_boundary_025_is_neutral(self, fuse):
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
#  7. NLP engine
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
        assert label == "Neutral"
        assert conf  == 1.0

    def test_whitespace_only_returns_neutral(self, nlp):
        label, _ = nlp(["   ", "\t", "\n"])
        assert label == "Neutral"

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
#  8. Caption → Sentiment integration
#  Tests that captions correctly influence both
#  the dashboard sentiment (via fusion) and the
#  Ollama prompt (via raw nlp_label passing).
# ══════════════════════════════════════════════

class TestCaptionSentiment:

    # ── No captions → expression drives everything ─────────────────────────

    def test_no_captions_neutral_expression_is_neutral(self):
        """No captions + neutral expression → neutral sentiment for dashboard."""
        result = _force_flush("Neutral", "No Gesture", "Person Center", captions=None)
        assert result is not None
        _, _, _, sentiment, _, _ = result
        assert sentiment == "neutral", \
            f"No captions + neutral expression should be neutral, got {sentiment}"

    def test_no_captions_smiling_tends_positive(self):
        """No captions + smiling expression → sentiment driven by expression polarity."""
        result = _force_flush("Smiling", "No Gesture", "Person Center", captions=None)
        assert result is not None
        _, _, _, sentiment, conf, description = result
        assert sentiment in ("positive", "neutral"), \
            f"No captions + smiling should be positive/neutral, got {sentiment}"
        assert isinstance(description, str) and len(description) > 0

    def test_no_captions_frowning_tends_negative(self):
        """No captions + frowning expression → sentiment driven by expression polarity."""
        result = _force_flush("Frowning", "No Gesture", "Person Center", captions=None)
        assert result is not None
        _, _, _, sentiment, conf, description = result
        assert sentiment in ("negative", "neutral"), \
            f"No captions + frowning should be negative/neutral, got {sentiment}"

    # ── Captions present → NLP reaches dashboard and prompt ───────────────

    def test_positive_captions_neutral_expression_produces_positive(self):
        """Strong positive captions + neutral face → positive dashboard sentiment.
        NLP (60% weight) dominates when expression is neutral."""
        result = _force_flush(
            "Neutral", "No Gesture", "Person Center",
            captions=[
                "This is a great idea, I love it!",
                "Absolutely brilliant, let's go with this.",
                "Yes, I fully agree.",
            ]
        )
        assert result is not None
        _, _, _, sentiment, conf, _ = result
        assert sentiment == "positive", \
            f"Strong positive captions + neutral face should be positive, got {sentiment} ({conf:.2f})"

    def test_negative_captions_neutral_expression_produces_negative(self):
        """Strong negative captions + neutral face → negative dashboard sentiment."""
        result = _force_flush(
            "Neutral", "No Gesture", "Person Center",
            captions=[
                "I completely disagree with this.",
                "This is not working at all.",
                "I'm very unhappy with this outcome.",
            ]
        )
        assert result is not None
        _, _, _, sentiment, conf, _ = result
        assert sentiment == "negative", \
            f"Strong negative captions + neutral face should be negative, got {sentiment} ({conf:.2f})"

    # ── Conflict: signals disagree → dashboard neutral, prompt gets conflict ──

    def test_conflict_smiling_negative_captions_neutral_on_dashboard(self):
        """Smiling + negative captions → fused sentiment is neutral for dashboard.
        But Ollama prompt should contain the conflict note (tested in TestPromptBuilding)."""
        result = _force_flush(
            "Smiling", "No Gesture", "Person Center",
            captions=[
                "I really disagree with this.",
                "This isn't working at all.",
                "I'm very unhappy with this outcome.",
            ]
        )
        assert result is not None
        _, _, _, sentiment, conf, description = result
        assert sentiment == "neutral", \
            f"Conflicting signals should produce neutral dashboard sentiment, got {sentiment}"
        # Description should still be generated
        assert isinstance(description, str) and len(description) > 0

    def test_conflict_frowning_positive_captions_neutral_on_dashboard(self):
        """Frowning + positive captions → fused sentiment is neutral for dashboard."""
        result = _force_flush(
            "Frowning", "No Gesture", "Person Center",
            captions=[
                "This is a great idea!",
                "I love this plan.",
                "Absolutely brilliant.",
            ]
        )
        assert result is not None
        _, _, _, sentiment, _, _ = result
        assert sentiment == "neutral", \
            f"Conflicting signals (frowning + positive captions) should be neutral, got {sentiment}"

    # ── Aligned signals boost confidence ──────────────────────────────────

    def test_aligned_signals_higher_confidence_than_expression_alone(self):
        """Smiling + positive captions should have higher dashboard confidence
        than smiling with no captions."""
        result_aligned = _force_flush(
            "Smiling", "No Gesture", "Person Center",
            captions=["This is great!", "Absolutely love it."]
        )
        result_expr_only = _force_flush(
            "Smiling", "No Gesture", "Person Center",
            captions=None
        )
        assert result_aligned   is not None
        assert result_expr_only is not None

        conf_aligned   = result_aligned[4]
        conf_expr_only = result_expr_only[4]

        assert conf_aligned >= conf_expr_only, \
            f"Aligned signals should have >= confidence: {conf_aligned:.2f} vs {conf_expr_only:.2f}"

    # ── Caption presence flag ──────────────────────────────────────────────

    def test_empty_list_treated_same_as_none(self):
        """captions=[] and captions=None should both mean 'no speech signal'."""
        result_none  = _force_flush("Neutral", "No Gesture", "Person Center", captions=None)
        result_empty = _force_flush("Neutral", "No Gesture", "Person Center", captions=[])
        assert result_none  is not None
        assert result_empty is not None
        # Both should produce neutral sentiment with no speech signal
        assert result_none[3]  == "neutral"
        assert result_empty[3] == "neutral"

    def test_description_generated_regardless_of_captions(self):
        """Description should always be a non-empty string whether or not
        captions are present."""
        result_with    = _force_flush("Smiling", "No Gesture", "Person Center",
                                      captions=["Great meeting."])
        result_without = _force_flush("Smiling", "No Gesture", "Person Center",
                                      captions=None)
        assert result_with    is not None and isinstance(result_with[5],    str)
        assert result_without is not None and isinstance(result_without[5], str)
        assert len(result_with[5].strip())    > 0
        assert len(result_without[5].strip()) > 0

    def test_run_system_source_mentions_summary(self):
        """The `run_system` implementation should include the word
        ``"summary"`` when constructing the payload sent to clients.  This
        guards against regressions where the key is accidentally removed.
        """
        import inspect, main

        src = inspect.getsource(main.run_system)
        assert '"summary"' in src

    # ── NLP label correctly reaches summarize() ────────────────────────────

    def test_caption_nlp_label_reaches_final_sentiment(self):
        """With neutral expression, NLP (60% weight) should dominate
        the fused sentiment."""
        result = _force_flush(
            "Neutral", "No Gesture", "Person Center",
            captions=[
                "This is absolutely wonderful.",
                "I'm so happy with this result.",
                "Best decision we've made.",
            ]
        )
        assert result is not None
        _, _, _, sentiment, conf, _ = result
        assert sentiment == "positive", \
            f"With neutral expression, positive captions should dominate, got {sentiment}"

    @pytest.mark.parametrize("captions,expected", [
        (
            ["Great work!", "Love this idea.", "Absolutely perfect."],
            "positive"
        ),
        (
            ["Terrible outcome.", "I hate this.", "This is a disaster."],
            "negative"
        ),
    ])
    def test_clear_captions_produce_correct_nlp_label(self, nlp, captions, expected):
        """NLP engine should classify clearly positive/negative captions correctly."""
        label, conf = nlp(captions)
        assert label == expected, \
            f"Expected {expected} for {captions}, got {label} ({conf:.2f})"
        assert conf >= 0.65, \
            f"Clear {expected} captions should have conf >= 0.65, got {conf:.2f}"


# ══════════════════════════════════════════════
#  9. getDominant
# ══════════════════════════════════════════════

class TestGetDominant:

    def test_empty_buffer_returns_neutral(self, dominant):
        assert dominant([], neutral="Neutral") == "Neutral"

    def test_single_weak_item_neutral_wins(self, dominant):
        result = dominant(
            [("Smiling", 0.80), ("Neutral", 0.80)],
            neutral="Neutral"
        )
        assert result == "Neutral", f"Got {result}"

    def test_single_item_no_neutral_wins_outright(self, dominant):
        result = dominant([("Smiling", 0.95)], neutral="Neutral")
        assert result == "Smiling", f"Got {result}"

    def test_strong_accumulated_score_wins(self, dominant):
        result = dominant([("Smiling", 0.95)] * 4, neutral="Neutral")
        assert result == "Smiling"

    def test_all_neutral(self, dominant):
        result = dominant([("Neutral", 1.0)] * 10, neutral="Neutral")
        assert result == "Neutral"

    def test_tie_returns_valid_label(self, dominant):
        result = dominant([("Smiling", 1.0), ("Frowning", 1.0)], neutral="Neutral")
        assert result in ("Smiling", "Frowning", "Neutral")


# ══════════════════════════════════════════════
#  10. Caption accumulation
# ══════════════════════════════════════════════

class TestCaptionAccumulation:

    def test_accumulates_across_frames(self):
        accumulated = []
        accumulated.extend(["This is great."])
        accumulated.extend(["I agree with this."])
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
#  11. Full flush cycle
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

    def test_no_captions_neutral_expression_is_neutral(self):
        result = _force_flush("Neutral", "No Gesture", "Person Center", captions=None)
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

    def test_empty_list_captions_doesnt_crash(self):
        import src.processor as p
        p.processExpression("Neutral", 1.0)
        p.processGesture("No Gesture", 1.0)
        p.processBodyAction("Person Center", 1.0)
        p._next_flush_time = 0
        result = p.flushAll(captions=[])
        assert result is not None

    def test_empty_buffers_return_neutral_defaults(self):
        import src.processor as p
        p._expr_buffer.clear()
        p._gest_buffer.clear()
        p._action_buffer.clear()
        p._next_flush_time = 0
        result = p.flushAll(captions=None)
        assert result is not None
        assert result[0] == "Neutral",       f"Got {result[0]}"
        assert result[1] == "No Gesture",    f"Got {result[1]}"
        assert result[2] == "Person Center", f"Got {result[2]}"

    def test_respects_interval_timer(self):
        import src.processor as p
        import time
        p._next_flush_time = time.time() + 999
        result = p.flushAll(captions=None)
        assert result is None, "Should return None before interval elapses"


# ══════════════════════════════════════════════
#  12. Overall confidence calculation
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
#  13. Confidence vs description consistency
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


# ══════════════════════════════════════════════
#  14. Expression hierarchy
# ══════════════════════════════════════════════

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

    def test_mouth_open_beats_smiling(self, expr_dominant):
        buf = [
            ("Mouth Open", 0.90),
            ("Smiling",    0.85),
            ("Smiling",    0.85),
            ("Smiling",    0.85),
        ]
        assert expr_dominant(buf) == "Mouth Open"

    def test_wink_beats_smiling(self, expr_dominant):
        buf = [
            ("Left Wink", 0.90),
            ("Smiling",   0.85),
            ("Smiling",   0.85),
        ]
        assert expr_dominant(buf) == "Left Wink"

    def test_right_wink_beats_frowning(self, expr_dominant):
        buf = [
            ("Right Wink", 0.90),
            ("Frowning",   0.88),
            ("Frowning",   0.88),
        ]
        assert expr_dominant(buf) == "Right Wink"

    def test_eyebrows_raised_beats_smiling(self, expr_dominant):
        buf = [
            ("Eyebrows Raised", 0.85),
            ("Smiling",         0.90),
            ("Smiling",         0.90),
        ]
        assert expr_dominant(buf) == "Eyebrows Raised"

    def test_frowning_beats_smiling(self, expr_dominant):
        buf = [
            ("Frowning", 0.85),
            ("Smiling",  0.90),
            ("Smiling",  0.90),
        ]
        assert expr_dominant(buf) == "Frowning"

    def test_smiling_beats_neutral(self, expr_dominant):
        buf = [("Smiling", 0.90)] * 4
        assert expr_dominant(buf) == "Smiling"

    def test_high_conf_low_priority_vs_low_conf_high_priority(self, expr_dominant):
        buf = [
            ("Mouth Open", 0.70),
            ("Smiling",    0.95),
        ]
        assert expr_dominant(buf) == "Mouth Open"

    def test_many_smiling_eventually_beats_single_low_conf_mouth_open(self, expr_dominant):
        buf = [
            ("Mouth Open", 0.60),
            ("Smiling",    0.85),
            ("Smiling",    0.85),
            ("Smiling",    0.85),
        ]
        assert expr_dominant(buf) == "Smiling"

    def test_equal_weighted_score_returns_valid(self, expr_dominant):
        buf = [
            ("Frowning", 1.0),
            ("Smiling",  0.75),
            ("Smiling",  0.75),
        ]
        result = expr_dominant(buf)
        assert result in ("Frowning", "Smiling")

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
        buf = [(higher, 0.85), (lower, 0.85)]
        result = expr_dominant(buf)
        assert result == higher, \
            f"{higher} (p{higher_priority}) should beat {lower} (p{lower_priority}), got {result}"

    def test_single_expression_returns_itself(self, expr_dominant):
        for expr in ("Mouth Open", "Left Wink", "Eyebrows Raised", "Frowning", "Smiling"):
            buf = [(expr, 0.90)] * 4
            assert expr_dominant(buf) == expr

    def test_all_neutral_returns_neutral(self, expr_dominant):
        assert expr_dominant([("Neutral", 1.0)] * 10) == "Neutral"

    def test_empty_buffer_returns_neutral(self, expr_dominant):
        assert expr_dominant([]) == "Neutral"

    def test_left_and_right_wink_same_priority(self, expr_dominant):
        buf = [("Left Wink", 0.90), ("Right Wink", 0.70)]
        assert expr_dominant(buf) == "Left Wink"