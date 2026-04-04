"""
translator.py — SOVA Translation Engine

Translates description strings into the configured target language.
Uses deep-translator (Google Translate backend) — free, no API key required.

Supported locales: en, ar, fr, es

Install:  pip install deep-translator

Offline alternative: swap GoogleTranslator for argostranslate
(requires downloading language packs on first run, ~100 MB each).
"""

from __future__ import annotations


SUPPORTED_LANGUAGES: dict[str, dict] = {
    "en": {
        "name":        "English",
        "gt_code":     "en",
        "tts_lang":    "en-US",
        "macos_voice": "Samantha",
        "rtl":         False,
    },
    "ar": {
        "name":        "Arabic",
        "gt_code":     "ar",
        "tts_lang":    "ar-SA",
        "macos_voice": "Majed", 
        "rtl":         True,
    },
    "fr": {
        "name":        "French",
        "gt_code":     "fr",
        "tts_lang":    "fr-FR",
        "macos_voice": "Thomas",    
        "rtl":         False,
    },
    "es": {
        "name":        "Spanish",
        "gt_code":     "es",
        "tts_lang":    "es-ES",
        "macos_voice": "Mónica", 
        "rtl":         False,
    },
}

DEFAULT_LOCALE = "en"

_translator    = None       # GoogleTranslator instance
_active_locale = "en"       # tracks which locale the instance was built for


def _get_translator(target_locale: str):
    """Return a cached GoogleTranslator, rebuilding if the locale changed."""
    global _translator, _active_locale

    if _translator is not None and _active_locale == target_locale:
        return _translator

    try:
        from deep_translator import GoogleTranslator  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "[TRANSLATOR] deep-translator is not installed. "
            "Run: pip install deep-translator"
        ) from exc

    gt_code        = SUPPORTED_LANGUAGES[target_locale]["gt_code"]
    _translator    = GoogleTranslator(source="auto", target=gt_code)
    _active_locale = target_locale
    print(f"[TRANSLATOR] Translator ready → {SUPPORTED_LANGUAGES[target_locale]['name']} ({gt_code})")
    return _translator


# ── Public helpers ────────────────────────────────────────────────────────────

def is_supported(locale: str) -> bool:
    """Return True if the locale code is in SUPPORTED_LANGUAGES."""
    return locale in SUPPORTED_LANGUAGES


def get_tts_lang(locale: str) -> str:
    """Return the BCP-47 TTS locale string for the given locale code.
    Used by pyttsx3 on Windows/Linux for voice matching."""
    return SUPPORTED_LANGUAGES.get(locale, SUPPORTED_LANGUAGES[DEFAULT_LOCALE])["tts_lang"]


def get_macos_voice(locale: str) -> str:
    """Return the macOS `say -v` voice name for the given locale code.
    Falls back to Samantha (English) if the locale isn't found."""
    return SUPPORTED_LANGUAGES.get(locale, SUPPORTED_LANGUAGES[DEFAULT_LOCALE])["macos_voice"]


def is_rtl(locale: str) -> bool:
    """Return True if the locale is right-to-left."""
    return SUPPORTED_LANGUAGES.get(locale, {}).get("rtl", False)


def language_options() -> list[dict]:
    """
    Return a list of dicts suitable for populating a UI dropdown.
    Each dict has: { "code": str, "name": str, "rtl": bool }
    """
    return [
        {"code": code, "name": meta["name"], "rtl": meta["rtl"]}
        for code, meta in SUPPORTED_LANGUAGES.items()
    ]


# ── Core API ──────────────────────────────────────────────────────────────────

def translate(description: str, target_locale: str) -> str:
    """
    Translate *description* into *target_locale*.

    Returns the translated string, or the original description unchanged if:
      - target_locale is English (no-op)
      - target_locale is unsupported
      - the translation API fails for any reason (network error, empty response)

    This function is intentionally safe — it will never raise and will never
    return an empty string.
    """

    # Guard: nothing to do
    if not description or not description.strip():
        return description

    # Guard: no-op for English
    if target_locale == DEFAULT_LOCALE:
        return description

    # Guard: unsupported locale
    if not is_supported(target_locale):
        print(f"[TRANSLATOR] Unsupported locale '{target_locale}' — returning original.")
        return description

    # Translate
    try:
        translator = _get_translator(target_locale)
        result     = translator.translate(description)

        if not result or not result.strip():
            print("[TRANSLATOR] Empty result from API — returning original.")
            return description

        return result

    except Exception as exc:
        print(f"[TRANSLATOR] Translation failed ({type(exc).__name__}: {exc}) — returning original.")
        return description