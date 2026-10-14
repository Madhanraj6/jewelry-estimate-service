"""Confidence levels with stated reasons, plus the human-review policy.

Two rules drive everything here:

1. A confidence level without a reason is decoration. Every field therefore returns
   ``{"level": ..., "reason": ...}`` where the reason names the evidence behind it.
2. Degraded output is never presented as normal output. An unreadable photo, a catalog
   photo shared by several SKUs, or a vision model that disagrees with the scorer all
   force review rather than a confident-looking quote.
"""

from __future__ import annotations

import re

REVIEW_THRESHOLD = 0.70
HIGH_CONFIDENCE_SCORE = 0.85
COUNT_AGREEMENT_TOLERANCE = 1
#: Product identity and quote-critical attributes required before a catalog match can be
#: selected without review.
MIN_ATTRIBUTES_FOR_AUTO_SELECT = 4
REQUIRED_ATTRIBUTES_FOR_AUTO_SELECT = ("item_type", "metal", "purity")

TEXT_FIELDS = ("item_type", "metal", "purity", "gemstone")


def determined_attribute_count(specifications: dict) -> int:
    """Count how many attributes the brief actually pinned down."""
    known = sum(
        1
        for field in TEXT_FIELDS
        if str(specifications.get(field, "unknown")).strip().lower() not in {"", "unknown"}
    )
    if float(specifications.get("target_weight_grams") or 0.0) > 0:
        known += 1
    return known

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

WEIGHT_RANGES = {  # Plausible grams per item type; outside these needs human eyes.
    "ring": (0.5, 25.0),
    "earrings": (0.5, 30.0),
    "pendant": (1.0, 40.0),
    "chain": (3.0, 80.0),
    "necklace": (3.0, 120.0),
    "bracelet": (3.0, 90.0),
}

#: Why each extracted field ends up at its level, keyed by the condition that fired.
FIELD_REASONS = {
    "llm_value": "extracted directly from the brief by the configured model",
    "llm_value_from_image": "extracted by the configured model from the reference photo",
    "local_rules": "recovered by the conservative local parser after the model was unavailable",
    "defaulted_unknown": "the brief did not state this and it cannot be read reliably from a photo",
    "defaulted_zero": "no usable weight was given; recorded as 0 rather than guessed",
    "outside_plausible_range": "value falls outside the plausible range for this product type",
}


def expected_component_count(brief: str) -> int | None:
    """Read an explicitly declared component count, if the brief contains one."""
    text = brief.lower()
    words = "|".join(NUMBER_WORDS)
    match = re.search(
        rf"\b(\d+|{words})\s+(?:gemstones?|stones?|diamonds?|components?)\b", text
    )
    if not match:
        return None
    value = match.group(1)
    return int(value) if value.isdigit() else NUMBER_WORDS[value]


def image_contradicts_brief(brief: str, component_count: int | None) -> bool:
    """Flag a material count disagreement while allowing one contour of noise."""
    expected = expected_component_count(brief)
    if expected is None or component_count is None:
        return False
    return abs(expected - component_count) > COUNT_AGREEMENT_TOLERANCE


def weight_is_plausible(item_type: str, weight: float) -> bool:
    """Check a weight against the plausible band for the product type."""
    low, high = WEIGHT_RANGES.get(str(item_type).strip().lower(), (0.0, 500.0))
    return low <= float(weight) <= high


def assess_review(
    match_score: float,
    brief: str,
    component_count: int | None,
    image_metadata: dict | None = None,
    vision_verification: dict | None = None,
    image_error: str | None = None,
    specifications: dict | None = None,
) -> dict:
    """Return a confidence band, a review decision, and every reason behind them."""
    image_metadata = image_metadata or {}
    flags: list[str] = []
    contradiction = image_contradicts_brief(brief, component_count)
    if contradiction:
        flags.append(
            f"brief implies about {expected_component_count(brief)} small parts but the "
            f"image count was {component_count}"
        )
    if match_score < REVIEW_THRESHOLD:
        flags.append(
            f"best catalog match scored {match_score:.2f}, below the {REVIEW_THRESHOLD:.2f} "
            "auto-select threshold"
        )
    if specifications is not None:
        determined = determined_attribute_count(specifications)
        missing_required = [
            field
            for field in REQUIRED_ATTRIBUTES_FOR_AUTO_SELECT
            if str(specifications.get(field, "unknown")).strip().lower()
            in {"", "unknown"}
        ]
        if missing_required:
            flags.append(
                "the brief did not specify quote-critical attribute(s): "
                + ", ".join(missing_required)
                + "; the catalog candidate cannot supply unstated product details"
            )
        if determined < MIN_ATTRIBUTES_FOR_AUTO_SELECT:
            flags.append(
                f"the brief determined only {determined} of 5 attributes "
                f"(at least {MIN_ATTRIBUTES_FOR_AUTO_SELECT} are required), so the top "
                "match is a guess; a higher score cannot make it safe to auto-select"
            )
    if image_error:
        flags.append(f"supplied image could not be read: {image_error}")
    elif image_metadata.get("product_type_unresolved"):
        flags.append(
            "the image did not provide a reliable product type; OpenCV visual similarity "
            "is not a jewelry-category classifier, so no product match should be recommended"
        )
    elif image_metadata.get("used") and not image_metadata.get("image_can_discriminate"):
        flags.append(image_metadata.get("reason") or "the catalog photo cannot identify this SKU")
    elif image_metadata.get("reason"):
        flags.append(f"image evidence degraded: {image_metadata['reason']}")
    for flag in (vision_verification or {}).get("flags", []):
        flags.append(flag)

    if flags:
        confidence = "low"
    elif match_score >= HIGH_CONFIDENCE_SCORE:
        confidence = "high"
    else:
        confidence = "medium"

    return {
        "confidence": confidence,
        "needs_human_review": bool(flags),
        "auto_selected": not flags,
        "selection_status": "needs_human_review" if flags else "auto_selected",
        "review_flags": flags,
        "text_image_contradiction": contradiction,
        "expected_component_count": expected_component_count(brief),
    }


def _specification_confidence(
    specifications: dict, extraction_metadata: dict, image_was_used: bool
) -> dict:
    """Grade each extracted specification field and explain the level."""
    from_image = extraction_metadata.get("image_understanding_used", False)
    fallback = extraction_metadata.get("fallback_used", False)
    item_type = str(specifications.get("item_type", "unknown"))

    confidence: dict[str, dict] = {}
    for field, value in specifications.items():
        unknown = value == "unknown" or (field == "target_weight_grams" and value == 0)
        if unknown:
            level = "low"
            reason = (
                FIELD_REASONS["defaulted_zero"]
                if field == "target_weight_grams"
                else FIELD_REASONS["defaulted_unknown"]
            )
        elif fallback:
            level = "medium"
            reason = FIELD_REASONS["local_rules"]
        elif image_was_used and from_image and field in {"item_type", "gemstone"}:
            level = "high"
            reason = FIELD_REASONS["llm_value_from_image"]
        else:
            level = "high"
            reason = FIELD_REASONS["llm_value"]
        confidence[field] = {"level": level, "reason": reason}

    weight = float(specifications.get("target_weight_grams") or 0.0)
    if weight and not weight_is_plausible(item_type, weight):
        confidence["target_weight_grams"] = {
            "level": "low",
            "reason": FIELD_REASONS["outside_plausible_range"],
        }
    return confidence


def _image_confidence(image_estimate: dict | None, image_metadata: dict | None) -> dict:
    """Grade the visual evidence as a whole."""
    image_metadata = image_metadata or {}
    result: dict = {
        "level": "not_available",
        "reason": "no reference photo was supplied, so nothing was counted or matched visually",
    }
    if image_estimate is None:
        return result
    if image_metadata.get("used") and not image_metadata.get("image_can_discriminate"):
        return {"level": "low", "reason": image_metadata.get("reason")}
    if image_metadata.get("candidates_with_shared_image"):
        shared = ", ".join(image_metadata["candidates_with_shared_image"])
        return {
            "level": "medium",
            "reason": (
                f"the catalog photo confirms the product type, but {shared} share one photo "
                "and cannot be told apart visually"
            ),
        }
    if image_metadata.get("used"):
        return {
            "level": "medium",
            "reason": (
                "photo compared against catalog photos with distinct imagery using "
                f"{image_metadata.get('backend')} embeddings; colour separation is approximate"
            ),
        }
    return {
        "level": "low",
        "reason": image_metadata.get("reason") or "image similarity could not be computed",
    }


def field_confidence(
    specifications: dict,
    extraction_metadata: dict,
    match_score: float,
    image_estimate: dict | None,
    image_error: str | None,
    image_metadata: dict | None = None,
    vision_verification: dict | None = None,
    needs_human_review: bool = False,
) -> dict:
    """Return a confidence level and a written reason for every output field."""
    if match_score >= HIGH_CONFIDENCE_SCORE:
        match_level, match_reason = "high", "top candidate scored at or above 0.85"
    elif match_score >= REVIEW_THRESHOLD:
        match_level, match_reason = "medium", "top candidate cleared 0.70 but not 0.85"
    else:
        match_level, match_reason = "low", "top candidate did not clear the 0.70 auto-select threshold"

    if needs_human_review:
        match_level = "low"
        match_reason = (
            "the top catalog candidate is a similarity match only and is held for human "
            "review; its score is not a probability that the item or price is correct"
        )

    return {
        "specifications": _specification_confidence(
            specifications, extraction_metadata, bool(image_estimate)
        ),
        "inventory_match": {"level": match_level, "reason": match_reason},
        "price": {
            "level": match_level,
            "reason": (
                "the catalog base price is not a confirmed custom quote; "
                + match_reason
            ),
        },
        "image_estimate": _image_confidence(image_estimate, image_metadata),
    }