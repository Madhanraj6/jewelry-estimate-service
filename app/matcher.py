"""Deterministic inventory matching, including image-to-catalog similarity.

No language model participates in this module. Selecting a SKU decides what a customer
is quoted and what gets shipped, so the score must be reproducible and auditable: the
same brief and the same photo always produce the same ranking, and every row's score can
be recomputed by hand from the documented weights below.

Why not ask the LLM to pick the SKU? Three reasons:

1. **Reproducibility.** A model can return a different ranking for identical input,
   which makes it impossible to explain a quote to a salesperson or to a regulator.
2. **Hallucination.** The SKU and its price must exist in ``data/invent.csv``. A model
   asked to "suggest the best SKU" can invent one, and invented prices are invisible
   unless every output is checked back against the catalog.
3. **Auditability.** Weighted scoring produces a number a human can re-derive from the
   inventory file. A generated answer has to be taken on trust.

Scoring is additive over independent attributes with fixed weights that sum to 1.0, so
scores stay inside [0, 1] and the auto-select threshold keeps a fixed meaning.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:  # Supports both `python -m app.matcher` and `python app/matcher.py`.
    from .embeddings import (
        EmbeddingError,
        cosine_similarity,
        embed_image,
        image_is_unique,
        inventory_image_path,
        load_manifest,
    )
    from .schemas import ExtractedSpecifications
except ImportError:
    from embeddings import (  # type: ignore[no-redef]
        EmbeddingError,
        cosine_similarity,
        embed_image,
        image_is_unique,
        inventory_image_path,
        load_manifest,
    )
    from schemas import ExtractedSpecifications  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

INVENTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "invent.csv"

#: Attribute weights. They sum to 1.0; see ``EVALUATION.md`` for the rationale.
#:
#: ``image_similarity`` compares the customer's photo against catalog photos directly.
#: ``semantic_similarity`` comes from the Qdrant index, where a vision model has already
#: turned each catalog photo into text, so it carries product-type and style meaning that
#: raw pixels do not. They are separate signals and are weighted separately: the vector
#: store widens the candidate set, it does not get to decide the price.
DEFAULT_WEIGHTS: dict[str, float] = {
    "item_type": 0.18,
    "metal": 0.15,
    "purity": 0.09,
    "gemstone": 0.08,
    "weight_grams": 0.05,
    "image_similarity": 0.20,
    "semantic_similarity": 0.25,
}

AUTO_SELECT_THRESHOLD = 0.70
REVIEW_THRESHOLD = 0.70
GEMSTONE_TOKEN_WEIGHT = 0.5  # Partial credit when "black onyx" meets "onyx".


def load_inventory() -> pd.DataFrame:
    """Load the local source of truth for catalog matching."""
    return pd.read_csv(INVENTORY_PATH, dtype={"sku": str}).fillna("")


def exact_match(expected: str, actual: str) -> float:
    """Return 1.0 for a case-insensitive exact match; otherwise 0.0."""
    return float(str(expected).strip().lower() == str(actual).strip().lower())


def gemstone_similarity(expected: str, actual: str) -> float:
    """Score two gemstone labels, allowing partial credit on shared words.

    A brief saying "black onyx" is a legitimate match for a catalog row tagged
    "black onyx", but should score below an exact hit so an exact row still wins.
    """
    left = str(expected).strip().lower()
    right = str(actual).strip().lower()
    if left == right:
        return 1.0
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    if not overlap:
        return 0.0
    jaccard = len(overlap) / len(left_tokens | right_tokens)
    return round(GEMSTONE_TOKEN_WEIGHT * jaccard, 6)


def weight_similarity_score(
    target_weight: float,
    inventory_weight: float,
    tolerance_grams: float = 5.0,
) -> float:
    """Return a linear 0.0-1.0 similarity score for two weights."""
    difference = abs(target_weight - inventory_weight)
    return max(0.0, 1 - (difference / tolerance_grams))


def attribute_scores(specs: ExtractedSpecifications, item: pd.Series) -> dict[str, float]:
    """Return every text/number attribute score for one inventory row."""
    return {
        "item_type": exact_match(specs.item_type, item["item_type"]),
        "metal": exact_match(specs.metal, item["metal"]),
        "purity": exact_match(specs.purity, item["purity"]),
        "gemstone": gemstone_similarity(specs.gemstone, item["gemstone"]),
        "weight_grams": weight_similarity_score(
            specs.target_weight_grams, float(item["weight_grams"])
        ),
    }


def _active_weights(has_image: bool, has_semantic: bool) -> dict[str, float]:
    """Return weights, redistributing the inactive shares across the live attributes.

    A text-only request with no index available must not silently keep a 0.45 share for
    two signals it does not have, so every unused weight is renormalized away.
    """
    if has_image and has_semantic:
        return dict(DEFAULT_WEIGHTS)

    active = {
        name: weight
        for name, weight in DEFAULT_WEIGHTS.items()
        if (name != "image_similarity" or has_image)
        and (name != "semantic_similarity" or has_semantic)
    }
    total = sum(active.values()) or 1.0
    weights = {name: weight / total for name, weight in active.items()}
    for name in DEFAULT_WEIGHTS:
        weights.setdefault(name, 0.0)
    return weights


def _image_similarity_column(
    inventory: pd.DataFrame, query_image: str | Path | bytes | None
) -> tuple[np.ndarray | None, dict]:
    """Cosine similarity of the query photo against every catalog photo.

    Raw cosines are min-max normalized across the candidate set before scoring. Raw
    values are heavily compressed near the top (most jewelry photographs look alike), so
    an absolute cosine would contribute almost no discrimination; relative normalization
    spreads the candidates across the full range regardless of backend.
    """
    metadata = {
        "used": False,
        "backend": None,
        "reason": None,
        "query_cached": None,
        "missing_catalog_images": [],
    }
    if not query_image:
        return None, metadata

    try:
        query_vector, query_meta = embed_image(query_image)
    except EmbeddingError as exc:
        metadata["reason"] = f"query image could not be embedded: {exc}"
        logger.warning("%s", metadata["reason"])
        return None, metadata
    except Exception as exc:
        metadata["reason"] = f"query image embedding failed: {type(exc).__name__}: {exc}"
        logger.warning("%s", metadata["reason"])
        return None, metadata

    metadata["used"] = True
    metadata["backend"] = query_meta["backend"]
    metadata["query_cached"] = query_meta["cached"]

    vectors: list[np.ndarray | None] = []
    for sku in inventory["sku"]:
        path = inventory_image_path(sku)
        if not path.is_file():
            metadata["missing_catalog_images"].append(sku)
            vectors.append(None)
            continue
        try:
            vector, _ = embed_image(path, backend=query_meta["backend"])
            vectors.append(vector)
        except Exception as exc:
            # A provider failure on one catalog photo must not abort the whole ranking.
            logger.warning("catalog image %s could not be embedded: %s", sku, exc)
            vectors.append(None)

    present = [i for i, v in enumerate(vectors) if v is not None]
    present_set = set(present)
    if len(present) < 2:
        metadata["used"] = False
        metadata["reason"] = "fewer than two catalog images could be embedded"
        return None, metadata

    raw = np.array(
        [cosine_similarity(query_vector, vectors[i]) if vectors[i] is not None else np.nan for i in range(len(vectors))]
    )
    window = raw[present]
    low, high = float(np.min(window)), float(np.max(window))
    span = high - low
    normalized = np.zeros(len(vectors), dtype=float)
    if span > 1e-9:
        for i in present:
            normalized[i] = (raw[i] - low) / span
    else:
        for i in present:
            normalized[i] = 0.5
        metadata["reason"] = "all catalog images scored identically; neutral image score applied"

    # Rows whose photo could not be embedded get the candidate-set mean, not a zero.
    for i in range(len(vectors)):
        if i not in present_set:
            normalized[i] = float(np.mean(normalized[present]))
    metadata["raw_cosine_min"] = round(low, 6)
    metadata["raw_cosine_max"] = round(high, 6)
    return normalized, metadata


def semantic_gate(specs: ExtractedSpecifications, item: pd.Series) -> float:
    """Whether visual similarity may contribute to this row at all.

    Image and semantic similarities promote a *plausible* product; neither may rescue
    one that contradicts what the brief explicitly stated. Only attributes the brief
    actually determined are enforced, so a vague brief keeps the full visual signal.
    """
    item_type = str(specs.item_type).strip().lower()
    metal = str(specs.metal).strip().lower()
    if item_type not in {"", "unknown"} and item_type != str(item["item_type"]).strip().lower():
        return 0.0
    if metal not in {"", "unknown"} and metal != str(item["metal"]).strip().lower():
        return 0.0
    return 1.0


def calculate_match_score(
    specs: ExtractedSpecifications,
    item: pd.Series,
    image_score: float = 0.0,
    semantic_score: float = 0.0,
    has_image: bool = False,
    has_semantic: bool = False,
) -> float:
    """Score one inventory row against the extracted specs.

    ``image_score`` and ``semantic_score`` are normalized to [0, 1]. The two boolean
    flags must be supplied by the caller rather than inferred from the scores: every row
    in a ranking has to be scored with the same weights, including the row that happens
    to land on the minimum.
    """
    weights = _active_weights(has_image=has_image, has_semantic=has_semantic)
    scores = attribute_scores(specs, item)
    total = sum(scores[name] * weights[name] for name in scores)
    return (
        total
        + weights["image_similarity"] * image_score * semantic_gate(specs, item)
        + weights["semantic_similarity"] * semantic_score * semantic_gate(specs, item)
    )


def _candidates(inventory: pd.DataFrame, scores: list[float], limit: int) -> list[dict]:
    frame = inventory.copy()
    frame["match_score"] = scores
    # SKU ascending breaks ties, so equal scores never reorder between runs.
    ranked = frame.sort_values(["match_score", "sku"], ascending=[False, True], kind="stable")
    candidates = []
    for rank, (_, item) in enumerate(ranked.iterrows(), start=1):
        score = round(float(item["match_score"]), 4)
        sku = item["sku"]
        candidates.append(
            {
                "rank": rank,
                "sku": sku,
                "item_type": item["item_type"],
                "metal": item["metal"],
                "purity": item["purity"],
                "gemstone": item["gemstone"],
                "weight_grams": float(item["weight_grams"]),
                "base_price": float(item["base_price"]),
                "image_url": item.get("image_url", ""),
                "image_is_unique": image_is_unique(sku),
                "match_score": score,
                "auto_selected": score >= AUTO_SELECT_THRESHOLD,
            }
        )
    return candidates[:limit]


def find_top_matches(
    specs: ExtractedSpecifications,
    limit: int = 3,
    query_image: str | Path | bytes | None = None,
    semantic_scores: dict[str, float] | None = None,
) -> tuple[list[dict], dict]:
    """Rank the catalog against the extracted specs, an optional photo, and the index.

    ``semantic_scores`` maps SKU to a normalized [0, 1] vector-store score. SKUs absent
    from the mapping get 0.0, so a partial vector recall degrades smoothly instead of
    silently dropping those rows.

    Returns the ranked candidates plus metadata describing how image and vector evidence
    were used, so the caller can report each contribution honestly.
    """
    inventory = load_inventory()
    image_scores, image_metadata = _image_similarity_column(inventory, query_image)
    semantic_scores = semantic_scores or {}

    scores: list[float] = []
    has_image = image_scores is not None
    has_semantic = bool(semantic_scores)
    for position, (_, item) in enumerate(inventory.iterrows()):
        image_score = 0.0 if image_scores is None else float(image_scores[position])
        semantic_score = float(semantic_scores.get(item["sku"], 0.0))
        scores.append(
            calculate_match_score(
                specs, item, image_score, semantic_score,
                has_image=has_image, has_semantic=has_semantic,
            )
        )

    candidates = _candidates(inventory, scores, limit)

    unique = [c["sku"] for c in candidates if c["image_is_unique"]]
    shared = [c["sku"] for c in candidates if not c["image_is_unique"]]
    image_metadata["candidates_with_unique_image"] = unique
    image_metadata["candidates_with_shared_image"] = shared
    # Only the top candidate's photo matters for the decision: if it is shared, the
    # image cannot confirm which of the SKUs using that photo was meant.
    image_metadata["image_can_discriminate"] = bool(candidates) and candidates[0]["image_is_unique"]
    if not image_metadata["image_can_discriminate"] and image_metadata["used"]:
        manifest_entry = load_manifest().get(candidates[0]["sku"], {})
        image_metadata["reason"] = (
            f"the top candidate {candidates[0]['sku']} shares one catalog photo with "
            f"{len(manifest_entry.get('shared_with', []))} other SKU(s), so the image cannot "
            "identify this specific product"
        )
    image_metadata["semantic_used"] = has_semantic
    image_metadata["weights"] = _active_weights(has_image=has_image, has_semantic=has_semantic)
    return candidates, image_metadata


def find_best_match(
    specs: ExtractedSpecifications,
    query_image: str | Path | bytes | None = None,
    semantic_scores: dict[str, float] | None = None,
) -> dict:
    """Return the highest-scoring inventory item and its deterministic score."""
    return find_top_matches(
        specs, limit=1, query_image=query_image, semantic_scores=semantic_scores
    )[0][0]


if __name__ == "__main__":
    demo = ExtractedSpecifications(
        item_type="ring", metal="gold", purity="18K", gemstone="diamond", target_weight_grams=3.2
    )
    for candidate in find_top_matches(demo)[0]:
        print(candidate)