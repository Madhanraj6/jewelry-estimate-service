"""LLM-backed extraction of structured jewelry specifications.

The provider chain is configurable by environment variable and always ends in a
deterministic local parser, so a request never fails and a degraded answer is always
labelled as degraded rather than presented as model output.

``GEMINI_MODEL`` -> ``GEMINI_FALLBACK_MODEL`` -> local rules
``OPENAI_MODEL`` -> local rules (kept so an OpenAI key still works)

Every response carries ``extraction`` metadata: ``source``, ``fallback_used``,
``fallback_reason`` and ``attempts``.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Final

from .normalize import normalise_specifications
from .schemas import ExtractedSpecifications

logger = logging.getLogger(__name__)

DEFAULT_SPECIFICATIONS: Final[ExtractedSpecifications] = ExtractedSpecifications(
    item_type="unknown",
    metal="unknown",
    purity="unknown",
    gemstone="unknown",
    target_weight_grams=0.0,
)

SYSTEM_PROMPT: Final[str] = """Extract jewelry specifications from the customer's brief and optional photo.
Return the requested schema only. Normalize item types and metals to lowercase. Use
'unknown' for unavailable text fields and 0 for an unavailable weight. Do not invent
details that are not supported by the brief or image. A photo alone cannot establish
metal purity or exact weight, so leave those fields unknown or 0 unless text provides them."""

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
#: gemini-3.8-flash intermittently answers 503/429, so a second model is tried. Verified
#: available for a free-tier key; gemini-2.5-flash and gemini-2.5-flash-lite now 404.
DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-3.5-flash-lite"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def _model_chain() -> list[tuple[str, str]]:
    """Return (provider, model) pairs to try, most preferred first.

    Older defaults such as ``gemini-2.5-flash`` now return 404 for new accounts, and
    ``gemini-3.8-flash`` intermittently returns 503 under load, so the chain carries a
    second model by default rather than relying on retries against one endpoint.
    """
    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    chain: list[tuple[str, str]] = []
    if provider in {"auto", "gemini"}:
        primary = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        chain.append(("gemini", primary))
        fallback = os.getenv("GEMINI_FALLBACK_MODEL", DEFAULT_GEMINI_FALLBACK_MODEL)
        if fallback and fallback != primary:
            chain.append(("gemini", fallback))
    if provider in {"auto", "openai"}:
        chain.append(("openai", os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)))
    return chain


def extract_specifications(brief: str, max_attempts: int = 2) -> ExtractedSpecifications:
    """Backward-compatible convenience wrapper returning specifications only."""
    specifications, _ = extract_with_metadata(brief, max_attempts=max_attempts)
    return specifications


def extract_with_metadata(
    brief: str, image: str | None = None, max_attempts: int = 2
) -> tuple[ExtractedSpecifications, dict]:
    """Return validated specifications, falling back safely if the LLM is unavailable.

    Each configured model gets ``max_attempts`` tries. The first model that returns a
    schema-valid result wins; otherwise the local parser produces the answer and the
    metadata says so.
    """
    if (not brief or not brief.strip()) and not image:
        return DEFAULT_SPECIFICATIONS.model_copy(), {
            "fallback_used": True,
            "fallback_reason": "empty_brief",
            "attempts": 0,
            "source": "safe_default",
            "model": None,
            "image_understanding_used": False,
        }

    chain = [pair for pair in _model_chain() if os.getenv(_credential_env(pair[0]))]
    if not chain:
        logger.warning("no LLM credentials configured; using the local extraction fallback")
        return _fallback(brief, "no_api_key_configured")

    attempts = 0
    failures: list[str] = []
    for provider, model in chain:
        for _ in range(max(1, max_attempts)):
            attempts += 1
            try:
                if provider == "gemini":
                    specs = _gemini_extraction(model, brief, image)
                else:
                    specs = _openai_extraction(model, brief, image)
                # Snap onto catalog vocabulary before anything scores it: a value like
                # "purple stone" cannot match any row and would silently waste its share
                # of the match score while reading like a confident answer.
                specs = normalise_specifications(specs, brief)
                return specs, {
                    "fallback_used": False,
                    "fallback_reason": None,
                    "attempts": attempts,
                    "source": f"{provider}_structured_output",
                    "model": model,
                    "image_understanding_used": bool(image),
                }
            except Exception as exc:  # Provider, transport, and schema failures are retryable.
                failures.append(f"{model}: {type(exc).__name__}: {exc}")
                logger.warning("extraction via %s failed: %s", model, exc)

    reason = f"llm_request_failed_after_{attempts}_attempts"
    logger.error("extraction failed (%s); using local fallback", reason)
    specs, metadata = _fallback(brief, reason)
    metadata["failed_models"] = failures
    return specs, metadata


def _fallback(brief: str, reason: str) -> tuple[ExtractedSpecifications, dict]:
    return normalise_specifications(fallback_from_brief(brief), brief), {
        "fallback_used": True,
        "fallback_reason": reason,
        "attempts": 0,
        "source": "local_rules",
        "model": None,
        "image_understanding_used": False,
    }


def _credential_env(provider: str) -> str:
    return "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"


# ------------------------------------------------------------------------- providers


def _gemini_extraction(
    model: str, brief: str, image: str | None = None
) -> ExtractedSpecifications:
    """One schema-constrained Gemini call over the brief and optional photo."""
    from google.genai import types

    from .embeddings import gemini_client

    client = gemini_client()

    parts: list = [
        types.Part.from_text(
            text=brief
            or "Infer only visually supportable jewelry specifications from this image."
        )
    ]
    if image:
        raw, mime = _image_bytes(image)
        parts.append(types.Part.from_bytes(data=raw, mime_type=mime))

    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ExtractedSpecifications,
            temperature=0.0,
        ),
    )
    if not response.parsed:
        raise ValueError("Gemini returned no parsed structured output")
    return ExtractedSpecifications.model_validate(response.parsed)


def _openai_extraction(
    model: str, brief: str, image: str | None = None
) -> ExtractedSpecifications:
    """One schema-constrained OpenAI call; used when an OpenAI key is configured."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    user_content: str | list[dict] = brief or "Infer only visually supportable jewelry specifications from this image."
    if image:
        raw, mime = _image_bytes(image)
        encoded = base64.b64encode(raw).decode("ascii")
        user_content = [
            {"type": "text", "text": f"Customer brief: {brief or 'No text brief was supplied.'}"},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "auto"}},
        ]

    completion = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=ExtractedSpecifications,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("the model returned no parsed structured output")
    return ExtractedSpecifications.model_validate(parsed)


# -------------------------------------------------------------------------- images


def _image_bytes(image: str) -> tuple[bytes, str]:
    """Pass data URLs through, or read an evaluation file from disk."""
    if image.startswith("data:image/"):
        header, _, encoded = image.partition(",")
        mime = header.split(";")[0].removeprefix("data:")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ValueError("Image data URL is not valid base64") from exc
        if len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError("Image must be 5 MB or smaller")
        return raw, mime

    path = Path(image)
    if not path.is_file():
        raise ValueError("Image must be a valid data URL or readable local file path")
    raw = path.read_bytes()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Image must be 5 MB or smaller")
    return raw, mimetypes.guess_type(path.name)[0] or "image/jpeg"


def fallback_from_brief(brief: str) -> ExtractedSpecifications:
    """Conservatively extract common terms when the configured LLM is unavailable.

    This is deliberately narrow: unknown fields remain unknown rather than being
    guessed. It keeps local demos and evaluation reproducible without an API key.
    """
    text = brief.lower()

    def first_match(options: tuple[str, ...], default: str = "unknown") -> str:
        return next((option for option in options if re.search(rf"\b{re.escape(option)}\b", text)), default)

    item_type = first_match(("ring", "chain", "necklace", "bracelet", "earrings", "pendant"))
    metal = first_match(("rose gold", "gold", "silver", "platinum"))
    purity_match = re.search(r"\b(14k|18k|22k|925|950)\b", text)
    purity = purity_match.group(1).upper() if purity_match else "unknown"
    gemstone = first_match(
        ("diamond", "ruby", "emerald", "sapphire", "amethyst", "pearl", "turquoise", "onyx", "morganite", "moissanite"),
        default="none" if re.search(r"\b(no gemstone|plain|without stones)\b", text) else "unknown",
    )
    weight_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:g|grams?)\b", text)
    weight = float(weight_match.group(1)) if weight_match else 0.0

    return ExtractedSpecifications(
        item_type=item_type,
        metal=metal,
        purity=purity,
        gemstone=gemstone,
        target_weight_grams=weight,
    )