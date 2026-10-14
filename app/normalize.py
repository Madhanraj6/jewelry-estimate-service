"""Snap extracted values onto the catalog's own vocabulary.

A language model returns whatever phrasing the brief used: "18k", "sterling silver",
"white gold", "a purple stone", "titanium". None of those are catalog values, so they
score 0 against every row and quietly waste their share of the match score. Worse, a
value like "white gold" looks like a confident answer while being unable to match
anything.

This module maps those onto values that actually exist in ``data/invent.csv``, or onto
``unknown`` when there is no honest mapping. It is deterministic, runs after schema
validation, and never invents a value the brief did not support.
"""

from __future__ import annotations

import re

from .matcher import load_inventory
from .schemas import ExtractedSpecifications

#: Free-form metal names mapped onto the catalog's metal column.
METAL_SYNONYMS: dict[str, str] = {
    "yellow gold": "gold",
    "white gold": "gold",
    "rose": "rose gold",
    "pink gold": "rose gold",
    "sterling": "silver",
    "sterling silver": "silver",
    "silver": "silver",
    "gold": "gold",
    "platinum": "platinum",
    "platinum 950": "platinum",
    "950 platinum": "platinum",
    "titanium": "unknown",
    "steel": "unknown",
    "brass": "unknown",
    "bronze": "unknown",
}

#: Free-form stone names mapped onto the catalog's gemstone column.
GEMSTONE_SYNONYMS: dict[str, str] = {
    "amethyst": "amethyst",
    "purple stone": "amethyst",
    "ruby": "ruby",
    "red stone": "ruby",
    "emerald": "emerald",
    "green stone": "emerald",
    "diamond": "diamond",
    "sapphire": "sapphire",
    "blue stone": "sapphire",
    "pearl": "pearl",
    "turquoise": "turquoise",
    "morganite": "morganite",
    "onyx": "black onyx",
    "black onyx": "black onyx",
    "moissanite": "moissanite",
    "none": "none",
    "no gemstone": "none",
    "no stone": "none",
    "plain": "none",
    "stone": "unknown",
    "gemstone": "unknown",
}

#: Words that mean the brief asserted the absence of a stone.
NONE_HINTS = ("no gemstone", "no stones", "without stone", "plain", "stone free", "no gems")

ITEM_TYPE_SYNONYMS: dict[str, str] = {
    "pendant": "pendant",
    "necklace": "necklace",
    "chain": "chain",
    "bracelet": "bracelet",
    "bangle": "bracelet",
    "ring": "ring",
    "earrings": "earrings",
    "earring": "earrings",
    "stud": "earrings",
}


def catalog_values(column: str) -> set[str]:
    """Return the distinct values a catalog column actually uses."""
    inventory = load_inventory()
    return {str(v).strip().lower() for v in inventory[column] if str(v).strip()}


def _in_catalog(candidate: str, column: str) -> bool:
    """Case-insensitive membership test against a lowercased vocabulary set."""
    return candidate.strip().lower() in catalog_values(column)


def _normalise_purity(value: str) -> str:
    """Uppercase and canonicalise purity so it matches the catalog spelling."""
    text = str(value).strip()
    if not text or text.lower() == "unknown":
        return "unknown"
    karat = re.fullmatch(r"\s*(14|18|22|24)\s*k\s*", text, re.I)
    if karat:
        candidate = f"{karat.group(1)}K"
        return candidate if _in_catalog(candidate, "purity") else "unknown"
    fineness = re.fullmatch(r"\s*9\s*(25|50)\s*", text, re.I)
    if fineness:
        candidate = f"9{fineness.group(1)}"
        return candidate if _in_catalog(candidate, "purity") else "unknown"
    upper = text.upper().replace(" ", "")
    return upper if _in_catalog(upper, "purity") else "unknown"


def _normalise_lookup(value: str, synonyms: dict[str, str], column: str) -> str:
    """Map free text onto a catalog value, or ``unknown`` when there is no honest match."""
    text = str(value).strip().lower()
    if not text or text == "unknown":
        return "unknown"
    if text in synonyms:
        return synonyms[text]
    # "silver and gold" / "gold-plated ring" style values: try the longest known synonym
    # that appears as a whole word in the phrase.
    for phrase, mapped in sorted(synonyms.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            return mapped
    return text if text in catalog_values(column) else "unknown"


def _normalise_gemstone(value: str, brief: str) -> str:
    text = str(value).strip().lower()
    if not text or text == "unknown":
        # The model said "unknown" but the brief may still imply the absence of a stone.
        if any(hint in brief.lower() for hint in NONE_HINTS):
            return "none"
        return "unknown"
    if text in catalog_values("gemstone"):
        return text
    return _normalise_lookup(text, GEMSTONE_SYNONYMS, "gemstone")


def normalise_specifications(
    specs: ExtractedSpecifications, brief: str = ""
) -> ExtractedSpecifications:
    """Return a copy of ``specs`` with every field snapped to a catalog value."""
    item_type = _normalise_lookup(specs.item_type, ITEM_TYPE_SYNONYMS, "item_type")
    if item_type == "earring":
        item_type = "earrings"

    purity = _normalise_purity(specs.purity)
    metal = _normalise_lookup(specs.metal, METAL_SYNONYMS, "metal")
    gemstone = _normalise_gemstone(specs.gemstone, brief)

    weight = float(specs.target_weight_grams or 0.0)
    if weight <= 0:
        weight = 0.0
    elif weight > 5000:
        # Only values beyond any plausible product are discarded. A merely odd weight
        # such as 900 g is preserved, because the brief did say it: discarding it would
        # hide the sales rep's typo, and the review engine already flags weights outside
        # the plausible band for the product type.
        weight = 0.0

    return specs.model_copy(
        update={
            "item_type": item_type,
            "metal": metal,
            "purity": purity,
            "gemstone": gemstone,
            "target_weight_grams": weight,
        }
    )