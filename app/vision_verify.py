"""Optional vision-model re-check of the deterministic shortlist.

The matcher ranks candidates with arithmetic alone. This module asks a vision model a
narrower question afterwards: does the customer's photo plausibly show *this* product
type, in roughly *this* metal colour? The answer can only raise the risk of a wrong
match and force human review.

It deliberately cannot reorder or replace the ranking. Re-ranking on a model's opinion
would reintroduce the non-determinism that the deterministic matcher exists to avoid,
and it would let a confident-sounding generation quietly change what a customer is
quoted. Verification is separated from selection so the pricing path stays auditable.

Without ``GEMINI_API_KEY`` this returns ``used=False`` and the service falls back to the
matcher's own evidence.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Final

logger = logging.getLogger(__name__)

DEFAULT_VERIFY_MODEL = "gemini-3.8-flash"
DISAGREEMENT_MARGIN = 0.20

VERIFY_PROMPT: Final[str] = """You are checking a jewelry catalog match.

The customer supplied a reference photo and a sales brief. A deterministic scorer
proposed catalog candidates. For each candidate you are shown its catalog photo.

For every candidate, judge two things only:
1. product_type_match: does the catalog photo show the same product type as the
   customer's photo (ring, chain, necklace, bracelet, earrings, pendant)?
2. metal_colour_match: does the catalog photo show roughly the same metal colour as the
   customer's photo (gold, rose gold, silver, platinum)?

Do not judge gemstone, purity, weight, or price from the image. Do not guess. If you
cannot tell, say false and explain why in `note`."""


def _client():
    from .embeddings import gemini_client

    return gemini_client()


def _inline_image(raw: bytes, mime: str) -> dict:
    return {"mime_type": mime, "data": base64.b64encode(raw).decode("ascii")}


def _decode_response(text: str) -> dict:
    """Pull the JSON object out of a model reply, tolerating markdown fences."""
    import json

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned[4:] if cleaned.lower().startswith("json") else cleaned
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("verification reply contained no JSON object")
    return json.loads(cleaned[start : end + 1])


def verify_candidates(
    brief: str,
    query_image: str | Path | bytes | None,
    candidates: list[dict],
) -> dict:
    """Ask a vision model whether the shortlist is visually consistent with the photo.

    Returns a dict with ``used``, per-candidate verdicts, and ``flags``. Never raises:
    a verification failure degrades to "not used" so the request still completes.
    """
    result: dict = {"used": False, "model": None, "flags": [], "candidates": [], "reason": None}
    if not query_image or not candidates:
        result["reason"] = "no query image or no candidates to verify"
        return result
    if os.getenv("VISION_VERIFY_ENABLED", "1").strip().lower() in {"0", "false", "no"}:
        result["reason"] = "vision verification disabled by VISION_VERIFY_ENABLED"
        return result

    try:
        from app.embeddings import EmbeddingError, read_image_bytes

        raw, mime = read_image_bytes(query_image)
    except (ImportError, ValueError) as exc:
        result["reason"] = f"query image unusable: {exc}"
        return result

    try:
        from google.genai import types
    except ImportError:
        result["reason"] = "google-genai is not installed"
        return result

    model = os.getenv("GEMINI_VISION_MODEL", DEFAULT_VERIFY_MODEL)
    result["model"] = model

    parts: list[dict] = [{"text": f"Customer brief:\n{brief or '(no text brief)'}\n\nCustomer photo:"}]
    parts.append({"inline_data": _inline_image(raw, mime)})
    parts.append({"text": "Catalog candidates:"})
    for candidate in candidates:
        try:
            from app.embeddings import inventory_image_path

            candidate_raw, candidate_mime = read_image_bytes(inventory_image_path(candidate["sku"]))
        except (ImportError, ValueError, EmbeddingError) as exc:
            logger.warning("skipping %s in verification: %s", candidate["sku"], exc)
            continue
        parts.append(
            {
                "text": (
                    f"Candidate {candidate['rank']} (SKU {candidate['sku']}, "
                    f"{candidate['item_type']}, {candidate['metal']}):"
                )
            }
        )
        parts.append({"inline_data": _inline_image(candidate_raw, candidate_mime)})
    if len(parts) <= 4:
        result["reason"] = "no catalog images could be loaded for verification"
        return result

    parts.append(
        {
            "text": (
                'Reply with JSON only: {"candidates":[{"sku":"...",'
                '"product_type_match":true,"metal_colour_match":true,"note":"..."}]}'
            )
        }
    )

    try:
        response = _client().models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                system_instruction=VERIFY_PROMPT,
                temperature=0.0,
            ),
        )
        parsed = _decode_response(response.text or "")
    except Exception as exc:  # Provider errors must not fail the request.
        logger.warning("vision verification failed: %s", exc)
        result["reason"] = f"verification call failed: {exc}"
        return result

    by_sku = {c["sku"]: c for c in candidates}
    for verdict in parsed.get("candidates", []):
        sku = str(verdict.get("sku", ""))
        candidate = by_sku.get(sku)
        if candidate is None:
            continue
        product_ok = bool(verdict.get("product_type_match"))
        metal_ok = bool(verdict.get("metal_colour_match"))
        if not product_ok:
            result["flags"].append(f"vision model disputes the product type of {sku}")
        elif not metal_ok and candidate["rank"] == 1:
            result["flags"].append(f"vision model disputes the metal colour of the top match {sku}")
        candidate["vision_verdict"] = {
            "product_type_match": product_ok,
            "metal_colour_match": metal_ok,
            "note": str(verdict.get("note", ""))[:300],
        }

    if not result["flags"]:
        top = candidates[0]
        verdict = top.get("vision_verdict")
        if verdict and not verdict["metal_colour_match"]:
            result["flags"].append(
                f"vision model disagrees with the top match {top['sku']} on metal colour"
            )
    result["used"] = True
    result["candidates"] = [
        {"sku": c["sku"], **c["vision_verdict"]} for c in candidates if "vision_verdict" in c
    ]
    return result