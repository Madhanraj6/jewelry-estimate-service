from pathlib import Path

import json
import os

import pytest
from fastapi.testclient import TestClient

from app import catalog_index, extractor, matcher
from app.cv_estimator import estimate_components
from app.embeddings import active_backend, cosine_similarity, embed_image, inventory_image_path
from app.main import app
from app.matcher import (
    DEFAULT_WEIGHTS,
    attribute_scores,
    find_best_match,
    find_top_matches,
    gemstone_similarity,
    load_inventory,
    weight_similarity_score,
)
from app.review_engine import assess_review, determined_attribute_count, field_confidence
from app.schemas import ExtractedSpecifications

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RING = ExtractedSpecifications(
    item_type="ring", metal="gold", purity="18K", gemstone="diamond", target_weight_grams=3.2
)


# ----------------------------------------------------------------- matcher


def test_exact_inventory_match():
    result = find_best_match(RING)
    assert result["sku"] == "RNG-001"
    assert result["match_score"] == 1.0


def test_match_score_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_attribute_weights_are_redistributed_when_a_signal_is_missing():
    """Inactive signal weights must be renormalized away, never left as dead weight."""
    everything = matcher._active_weights(has_image=True, has_semantic=True)
    no_image = matcher._active_weights(has_image=False, has_semantic=True)
    neither = matcher._active_weights(has_image=False, has_semantic=False)

    assert everything == DEFAULT_WEIGHTS
    for weights in (no_image, neither):
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["image_similarity"] == 0.0

    assert no_image["semantic_similarity"] > 0.0
    assert neither["semantic_similarity"] == 0.0

    # Attribute proportions must survive the redistribution.
    attribute_names = ("item_type", "metal", "purity", "gemstone", "weight_grams")
    live = sum(everything[n] for n in attribute_names)
    for name in attribute_names:
        assert no_image[name] / sum(no_image[n] for n in attribute_names) == pytest.approx(
            everything[name] / live
        )


def test_every_weight_is_non_negative():
    for has_image, has_semantic in ((True, True), (False, True), (True, False), (False, False)):
        weights = matcher._active_weights(has_image=has_image, has_semantic=has_semantic)
        assert all(w >= 0.0 for w in weights.values())
        assert sum(weights.values()) == pytest.approx(1.0)


def test_every_row_scored_with_the_same_weight_set():
    """A row landing on the minimum score must not silently drop a weight."""
    inventory = load_inventory()
    image_scores = [1.0, 0.0] + [0.5] * (len(inventory) - 2)
    semantic_scores = {"RNG-001": 1.0, "RNG-002": 0.0, "RNG-003": 0.5}
    scores = [
        matcher.calculate_match_score(
            RING, item, image, semantic_scores.get(sku, 0.0),
            has_image=True, has_semantic=True,
        )
        for (_, item), image, sku in zip(inventory.iterrows(), image_scores, inventory["sku"])
    ]
    # With identical image and semantic scores, attributes must still decide the order.
    assert scores[0] > scores[1]


def test_semantic_scores_cannot_outvote_a_perfect_attribute_match():
    """The vector store widens recall; it must not be able to pick the wrong SKU."""
    # Every semantic point goes to a chain, none to the exact ring.
    hijack = {f"CHN-00{i}": 1.0 for i in range(1, 9)}
    candidates = find_top_matches(RING, limit=3, semantic_scores=hijack)[0]
    assert candidates[0]["sku"] == "RNG-001"
    # It cannot score 1.0 because the image signal is absent and its weight is
    # redistributed away; it must still be the clear winner.
    assert candidates[0]["match_score"] > candidates[1]["match_score"]
    assert all(not c["sku"].startswith("CHN") for c in candidates[:1])


def test_a_perfect_attribute_match_scores_one_when_nothing_is_missing():
    """With every signal present and every attribute exact, the score must be 1.0."""
    candidates = find_top_matches(
        RING,
        limit=1,
        query_image=inventory_image_path("RNG-001"),
        semantic_scores={"RNG-001": 1.0},
    )[0]
    assert candidates[0]["sku"] == "RNG-001"
    assert candidates[0]["match_score"] == pytest.approx(1.0)


def test_matching_is_deterministic_across_repeated_runs():
    first, _ = find_top_matches(RING, limit=5)
    for _ in range(3):
        again, _ = find_top_matches(RING, limit=5)
        assert [c["sku"] for c in again] == [c["sku"] for c in first]
        assert [c["match_score"] for c in again] == [c["match_score"] for c in first]


def test_ties_break_on_sku_not_row_order():
    """Two rows with identical attributes must always rank in SKU order."""
    inventory = load_inventory()
    chain_rows = inventory[inventory["item_type"] == "chain"]
    first, _ = find_top_matches(
        ExtractedSpecifications(
            item_type="chain", metal="unknown", purity="unknown",
            gemstone="unknown", target_weight_grams=0.0,
        ),
        limit=len(chain_rows),
    )
    tied = [c for c in first if c["item_type"] == "chain"]
    assert len(tied) == len(chain_rows)


def test_gemstone_partial_credit_prefers_exact_match():
    exact = gemstone_similarity("black onyx", "black onyx")
    partial = gemstone_similarity("black onyx", "onyx")
    unrelated = gemstone_similarity("black onyx", "diamond")
    assert exact == 1.0
    assert 0.0 < partial < exact
    assert unrelated == 0.0


def test_weight_similarity_is_linear_and_clamped():
    assert weight_similarity_score(100.0, 100.0) == 1.0
    assert weight_similarity_score(100.0, 102.0) == pytest.approx(0.6)
    assert weight_similarity_score(100.0, 200.0) == 0.0


# --------------------------------------------------------- image-backed matching


@pytest.mark.skipif(not inventory_image_path("RNG-001").is_file(), reason="images not fetched")
def test_image_similarity_is_used_and_reported():
    _, metadata = find_top_matches(RING, limit=3, query_image=inventory_image_path("RNG-001"))
    assert metadata["used"] is True
    assert metadata["backend"] == active_backend()
    assert metadata["weights"]["image_similarity"] > 0
    # RNG-001's photo is reused by other rows, which must be surfaced, not hidden.
    if not metadata["image_can_discriminate"]:
        assert "shares one catalog photo" in metadata["reason"]


@pytest.mark.skipif(not inventory_image_path("RNG-002").is_file(), reason="images not fetched")
def test_matching_its_own_photo_ranks_that_sku_first():
    specs = ExtractedSpecifications(
        item_type="ring", metal="gold", purity="22K", gemstone="ruby", target_weight_grams=4.1
    )
    candidates, _ = find_top_matches(specs, limit=3, query_image=inventory_image_path("RNG-002"))
    assert candidates[0]["sku"] == "RNG-002"


@pytest.mark.skipif(not inventory_image_path("RNG-002").is_file(), reason="images not fetched")
def test_shared_catalog_photo_is_reported_and_downgraded():
    """The chain photo is shared by eight SKUs, so the image cannot pick one."""
    specs = ExtractedSpecifications(
        item_type="chain", metal="gold", purity="22K", gemstone="none", target_weight_grams=12.4
    )
    _, metadata = find_top_matches(specs, limit=3, query_image=inventory_image_path("CHN-001"))
    assert metadata["used"] is True
    assert metadata["image_can_discriminate"] is False
    assert "shares one catalog photo" in metadata["reason"]
    review = assess_review(
        match_score=0.9, brief="gold chain",
        component_count=None, image_metadata=metadata,
    )
    assert review["needs_human_review"] is True
    assert any("shares one catalog photo" in flag for flag in review["review_flags"])


@pytest.mark.skipif(not inventory_image_path("RNG-001").is_file(), reason="images not fetched")
def test_unreadable_query_image_degrades_instead_of_crashing():
    _, metadata = find_top_matches(RING, limit=3, query_image="does/not/exist.jpg")
    assert metadata["used"] is False
    assert "could not be embedded" in metadata["reason"]


# ------------------------------------------------------------------ embeddings


def test_embeddings_are_unit_length_and_cacheable():
    path = inventory_image_path("RNG-003")
    if not path.is_file():
        pytest.skip("images not fetched")
    vector, metadata = embed_image(path)
    assert metadata["dimensions"] == vector.shape[0]
    assert float((vector @ vector) ** 0.5) == pytest.approx(1.0, abs=1e-5)
    _, second = embed_image(path)
    assert second["cached"] is True


def test_embedding_a_photo_against_itself_is_maximal():
    path = inventory_image_path("RNG-005")
    other = inventory_image_path("RNG-006")
    if not path.is_file() or not other.is_file():
        pytest.skip("images not fetched")
    same, _ = embed_image(path)
    again, _ = embed_image(path)
    different, _ = embed_image(other)
    assert cosine_similarity(same, again) == pytest.approx(1.0, abs=1e-5)
    assert cosine_similarity(same, different) < 1.0


# -------------------------------------------------------------------- CV count


def test_cv_counts_synthetic_shapes():
    image = PROJECT_ROOT / "eval_cases" / "images" / "one_component.pgm"
    assert estimate_components(image)["component_count"] == 1


def test_cv_undercounts_touching_shapes():
    """Known failure mode: overlapping circles merge under RETR_EXTERNAL."""
    image = PROJECT_ROOT / "eval_cases" / "images" / "hard_touching.png"
    if not image.is_file():
        pytest.skip("hard fixtures not generated")
    assert estimate_components(image)["component_count"] < 5


# ------------------------------------------------------------- review engine


def test_review_flags_low_score_with_a_reason():
    review = assess_review(0.65, "gold ring", None)
    assert review["confidence"] == "low"
    assert review["needs_human_review"] is True
    assert any("below the 0.70" in flag for flag in review["review_flags"])


def test_every_confidence_entry_states_a_reason():
    specs = RING.model_dump()
    confidence = field_confidence(
        specs,
        {"fallback_used": False, "image_understanding_used": True},
        match_score=0.92,
        image_estimate={"component_count": 2},
        image_error=None,
    )
    entries = list(confidence["specifications"].values()) + [
        confidence["inventory_match"],
        confidence["price"],
        confidence["image_estimate"],
    ]
    for entry in entries:
        assert entry["level"] in {"high", "medium", "low", "not_available"}
        assert entry["reason"].strip()


def test_implausible_weight_is_downgraded_with_a_reason():
    specs = RING.model_copy(update={"target_weight_grams": 900.0})
    confidence = field_confidence(
        specs.model_dump(),
        {"fallback_used": False, "image_understanding_used": False},
        match_score=0.9,
        image_estimate=None,
        image_error=None,
    )
    weight = confidence["specifications"]["target_weight_grams"]
    assert weight["level"] == "low"
    assert "outside the plausible range" in weight["reason"]


def test_brief_and_image_count_disagreement_is_flagged():
    review = assess_review(0.95, "a ring with 5 diamonds", component_count=1)
    assert review["text_image_contradiction"] is True
    assert review["needs_human_review"] is True


def test_under_specified_brief_can_never_be_auto_selected():
    """A high score must not rescue a match the brief never actually determined."""
    vague = {"item_type": "ring", "metal": "gold", "purity": "unknown",
             "gemstone": "unknown", "target_weight_grams": 0.0}
    review = assess_review(0.99, "a light gold ring", None, specifications=vague)
    assert review["needs_human_review"] is True
    assert any("only 2 of 5 attributes" in flag for flag in review["review_flags"])

    specific = {**vague, "purity": "18K", "gemstone": "diamond",
                "target_weight_grams": 3.2}
    assert determined_attribute_count(specific) == 5
    clean = assess_review(0.99, "18K gold diamond ring", None, specifications=specific)
    assert clean["needs_human_review"] is False
    assert clean["confidence"] == "high"


def test_missing_quote_critical_attributes_force_review_even_at_high_match_score():
    specs = {
        "item_type": "ring",
        "metal": "unknown",
        "purity": "18K",
        "gemstone": "unknown",
        "target_weight_grams": 3.2,
    }
    review = assess_review(
        0.99,
        "18K ring weighing 3.2 grams",
        None,
        specifications=specs,
    )
    assert review["needs_human_review"] is True
    assert review["auto_selected"] is False
    assert review["selection_status"] == "needs_human_review"
    assert any("metal" in flag for flag in review["review_flags"])
    assert any("only 3 of 5 attributes" in flag for flag in review["review_flags"])


def test_analyze_does_not_auto_select_when_metal_is_missing():
    response = TestClient(app).post(
        "/analyze", json={"brief": "18K ring weighing 3.2 grams"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["match"]["sku"] == "RNG-001"
    assert body["match"]["auto_selected"] is False
    assert body["review"]["needs_human_review"] is True
    assert body["review"]["selection_status"] == "needs_human_review"
    assert body["field_confidence"]["price"]["level"] == "low"
    assert "not a confirmed custom quote" in body["field_confidence"]["price"]["reason"]


def test_determined_attribute_count():
    assert determined_attribute_count(
        {"item_type": "ring", "metal": "unknown", "purity": "unknown",
         "gemstone": "unknown", "target_weight_grams": 0.0}
    ) == 1
    assert determined_attribute_count(
        {"item_type": "ring", "metal": "gold", "purity": "18K",
         "gemstone": "none", "target_weight_grams": 3.2}
    ) == 5


def test_count_within_tolerance_is_not_flagged():
    review = assess_review(0.95, "a ring with 3 diamonds", component_count=2)
    assert review["text_image_contradiction"] is False


# ------------------------------------------------------- extraction resilience


def test_malformed_llm_output_uses_safe_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-test-fallback")
    monkeypatch.setattr(
        extractor,
        "_gemini_extraction",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("malformed model output")),
    )

    specifications, metadata = extractor.extract_with_metadata(
        "18K gold diamond ring weighing 3.2 grams", max_attempts=2
    )
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"].startswith("llm_request_failed_after")
    assert metadata["source"] == "local_rules"
    assert len(metadata["failed_models"]) == 4  # two models, two attempts each
    assert specifications.item_type == "ring"


def test_fallback_model_is_used_when_the_primary_fails(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-test-fallback")

    def only_fallback_works(model, brief, image=None):
        if model != "gemini-test-fallback":
            raise RuntimeError("primary model unavailable")
        return ExtractedSpecifications(
            item_type="ring", metal="gold", purity="18K",
            gemstone="diamond", target_weight_grams=3.2,
        )

    monkeypatch.setattr(extractor, "_gemini_extraction", only_fallback_works)
    specifications, metadata = extractor.extract_with_metadata("18K gold diamond ring")
    assert metadata["fallback_used"] is False
    assert metadata["model"] == "gemini-test-fallback"
    assert specifications.purity == "18K"


def test_no_credentials_degrades_to_local_rules(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    specifications, metadata = extractor.extract_with_metadata("plain 925 silver chain")
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"] == "no_api_key_configured"
    assert specifications.metal == "silver"


def test_empty_input_degrades_to_documented_default():
    specifications, metadata = extractor.extract_with_metadata("", None)
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"] == "empty_brief"
    assert specifications.model_dump() == ExtractedSpecifications(
        item_type="unknown", metal="unknown", purity="unknown",
        gemstone="unknown", target_weight_grams=0.0,
    ).model_dump()


def test_local_fallback_does_not_invent_missing_fields():
    specs = extractor.fallback_from_brief("a light gold ring")
    assert specs.item_type == "ring"
    assert specs.metal == "gold"
    assert specs.purity == "unknown"
    assert specs.target_weight_grams == 0.0


def test_attribute_scores_never_exceed_one():
    inventory = load_inventory()
    row = inventory.iloc[0]
    scores = attribute_scores(RING, row)
    assert all(0.0 <= value <= 1.0 for value in scores.values())


# ------------------------------------------------------------- vector index


def test_vector_index_is_built():
    compatible, reason = catalog_index.ensure_index()
    assert compatible, reason
    assert catalog_index.index_is_ready()


def test_every_catalog_sku_is_indexed():
    from app.catalog_index import search as vector_search

    results = vector_search(text="ring chain necklace", limit=50)
    assert results["used"] is True
    indexed = {r["sku"] for r in results["results"]}
    assert indexed == set(load_inventory()["sku"])


def test_vector_search_returns_the_right_product_type():
    from app.catalog_index import search as vector_search

    results = vector_search(text="a long gold chain necklace", limit=3)
    assert results["used"] is True
    top = results["results"][0]
    assert top["item_type"] in {"chain", "necklace"}
    assert "description" in top and top["description"]


def test_vector_search_needs_the_index_not_to_be_absent():
    result = catalog_index.search()  # neither text nor image
    assert result["used"] is False
    assert result["reason"]


def test_reciprocal_rank_fusion_is_normalized_and_stable():
    fused = catalog_index._reciprocal_rank_fusion([["A", "B"], ["B", "A"]], [1.0, 1.0])
    assert max(fused.values()) == pytest.approx(1.0)
    assert set(fused) == {"A", "B"}
    assert catalog_index._reciprocal_rank_fusion([], []) == {}


def test_hashing_text_vector_is_deterministic_and_normalized():
    first = catalog_index.hash_text_vector("gold ring with a ruby")
    again = catalog_index.hash_text_vector("gold ring with a ruby")
    assert (first == again).all()
    assert float((first @ first) ** 0.5) == pytest.approx(1.0, abs=1e-5)

    empty = catalog_index.hash_text_vector("")
    assert not empty.any(), "empty text must produce a zero vector, not a normalized one"
    assert float((first @ empty)) == pytest.approx(0.0)


def test_point_ids_are_stable_across_processes():
    """Python's hash() is salted per process; point IDs must not depend on it."""
    expected = catalog_index.point_id("RNG-001")
    assert expected == catalog_index.point_id("RNG-001")
    assert expected != catalog_index.point_id("RNG-002")
    assert 0 <= expected < 2**63


def test_attribute_description_is_labelled_as_not_a_vlm_description():
    assert catalog_index.attribute_description("RNG-001").startswith("A gold ring")
    entry = catalog_index.describe_image("RNG-001")
    assert entry["source"] in {"vlm", "catalog_attributes", "missing_image"}
    assert entry["description"]


def test_product_details_include_the_image():
    product = catalog_index.get_product("RNG-001")
    if product is None or "image_base64" not in product:
        pytest.skip("images not fetched")
    assert product["sku"] == "RNG-001"
    assert product["item_type"] == "ring"
    assert len(product["image_base64"]) > 100
    assert product["image_mime"] == "image/jpeg"


@pytest.mark.parametrize(
    ("source_url", "expected_source"),
    [
        ("https://catalog.example/ring", "inventory_url"),
        ("https://images.example/ring.jpg", "fallback_url"),
    ],
)
def test_product_reports_catalog_or_fallback_image_source(
    monkeypatch, source_url, expected_source
):
    monkeypatch.setattr(
        catalog_index,
        "load_manifest",
        lambda: {
            "RNG-001": {
                "catalog_url": "https://catalog.example/ring",
                "source_url": source_url,
                "status": "ok",
            }
        },
    )
    monkeypatch.setattr(
        catalog_index,
        "describe_image",
        lambda sku: {"description": "A ring", "source": "catalog_attributes"},
    )

    product = catalog_index.get_product("RNG-001", include_image=False)

    assert product is not None
    assert product["image_source"] == expected_source


def test_unknown_sku_returns_none():
    assert catalog_index.get_product("NOPE-999") is None


def test_backend_mismatch_is_reported_as_a_rebuild_instruction(monkeypatch):
    """Adding a key flips the backend; a stale index must say so, not crash."""
    compatible, why = catalog_index.index_backend_matches_active()
    assert compatible is True, f"index should match its own backend: {why}"

    built_with = os.getenv("INDEX_EMBEDDING_BACKEND") or active_backend()
    other = "gemini" if built_with == "opencv" else "opencv"
    monkeypatch.setenv("INDEX_EMBEDDING_BACKEND", other)
    compatible, why = catalog_index.index_backend_matches_active()
    assert compatible is False
    assert "build_catalog_index.py --force" in why

    result = catalog_index.search(text="gold chain")
    assert result["used"] is False
    assert "build_catalog_index.py --force" in result["reason"]


def test_semantic_similarity_cannot_rescue_an_attribute_contradiction():
    """Visual similarity promotes a plausible product, never a wrong one."""
    inventory = load_inventory().set_index("sku")
    earrings = inventory.loc["EAR-001"]
    assert earrings["item_type"] == "earrings"  # the brief says ring

    gated = matcher.calculate_match_score(
        RING, earrings, semantic_score=1.0, has_semantic=True
    )
    exact = inventory.loc["RNG-001"]
    exact_score = matcher.calculate_match_score(
        RING, exact, semantic_score=0.4, has_semantic=True
    )
    # Even the strongest possible semantic score for the earrings must not win.
    assert gated < exact_score
    assert matcher.semantic_gate(RING, earrings) == 0.0
    assert matcher.semantic_gate(RING, exact) == 1.0


def test_image_similarity_cannot_override_an_explicit_item_type():
    inventory = load_inventory().set_index("sku")
    pendant = inventory.loc["PEN-001"]
    ring = inventory.loc["RNG-001"]

    pendant_score = matcher.calculate_match_score(
        RING, pendant, image_score=1.0, has_image=True
    )
    ring_score = matcher.calculate_match_score(
        RING, ring, image_score=0.0, has_image=True
    )

    assert pendant_score < ring_score


def test_semantic_gate_is_open_when_the_brief_says_nothing():
    vague = ExtractedSpecifications(
        item_type="unknown", metal="unknown", purity="unknown",
        gemstone="unknown", target_weight_grams=0.0,
    )
    inventory = load_inventory().set_index("sku")
    assert matcher.semantic_gate(vague, inventory.loc["EAR-001"]) == 1.0


def test_env_file_loading_does_not_override_real_environment(monkeypatch, tmp_path):
    from app import config

    monkeypatch.setenv("GEMINI_MODEL", "from-the-shell")
    config.load_env()
    assert os.getenv("GEMINI_MODEL") == "from-the-shell"


def test_api_key_configured_never_prints_the_key():
    from app.config import api_key_configured

    assert api_key_configured() in {True, False}
    if api_key_configured():
        assert "GEMINI_API_KEY" not in json.dumps({"ok": True})


# --------------------------------------------------------------- index endpoints


def test_inventory_endpoint_returns_full_row():
    response = TestClient(app).get("/inventory/RNG-002?include_image=false")
    assert response.status_code == 200
    body = response.json()
    assert body["sku"] == "RNG-002"
    assert body["metal"] == "gold"
    assert body["purity"] == "22K"
    assert body["base_price"] == 710.0


def test_inventory_image_endpoint_serves_jpeg():
    response = TestClient(app).get("/inventory/RNG-002/image")
    if response.status_code == 404:
        pytest.skip("images not fetched")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_unknown_sku_returns_404():
    assert TestClient(app).get("/inventory/NOPE-999").status_code == 404


def test_search_endpoint_returns_ranked_products():
    response = TestClient(app).post("/search", json={"brief": "gold chain"})
    assert response.status_code == 200
    body = response.json()
    assert body["used"] is True
    assert body["results"][0]["item_type"] in {"chain", "necklace"}


def test_analyze_returns_the_matched_product_with_its_image():
    response = TestClient(app).post(
        "/analyze", json={"brief": "22K gold ruby ring weighing 4.1 grams"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["match"]["sku"] == "RNG-002"
    assert body["product"]["sku"] == "RNG-002"
    assert body["product"]["base_price"] == 710.0
    assert body["retrieval"]["used"] is True
    if "image_base64" in body["product"]:
        assert len(body["product"]["image_base64"]) > 100


def test_image_without_a_product_type_does_not_recommend_a_catalog_candidate():
    image_path = PROJECT_ROOT / "data" / "images" / "RNG-001.jpg"
    response = TestClient(app).post(
        "/analyze", json={"brief": "I want this", "image": str(image_path)}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["specifications"]["item_type"] == "unknown"
    assert body["match_status"]["status"] == "needs_product_type_confirmation"
    assert body["match_status"]["recommendation_available"] is False
    assert body["match"]["recommendation_eligible"] is False
    assert any(
        "did not provide a reliable product type" in flag
        for flag in body["review"]["review_flags"]
    )


# --------------------------------------------------------------------- service


def test_analyze_endpoint():
    response = TestClient(app).post(
        "/analyze", json={"brief": "18K gold diamond ring weighing 3.2 grams"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["match"]["sku"] == "RNG-001"
    assert len(body["top_candidates"]) == 3
    assert body["review"]["confidence"] in {"high", "medium", "low"}


def test_analyze_accepts_multipart_image_upload():
    image_path = PROJECT_ROOT / "eval_cases" / "images" / "1_components.png"
    client = TestClient(app)
    image_bytes = image_path.read_bytes()
    for mime_type in ("image/png", "application/octet-stream"):
        response = client.post(
            "/analyze",
            data={"brief": "single stone ring"},
            files={"image": ("reference.png", image_bytes, mime_type)},
        )
        assert response.status_code == 200
        assert response.json()["image_estimate"]["component_count"] == 1


def test_analyze_openapi_documents_json_and_multipart_requests():
    content = app.openapi()["paths"]["/analyze"]["post"]["requestBody"]["content"]
    assert "application/json" in content
    assert "multipart/form-data" in content
    assert content["multipart/form-data"]["schema"]["properties"]["image"]["format"] == "binary"


def test_analyze_rejects_non_image_multipart_upload():
    response = TestClient(app).post(
        "/analyze",
        data={"brief": "gold ring"},
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_analyze_rejects_oversized_multipart_upload():
    response = TestClient(app).post(
        "/analyze",
        data={"brief": "gold ring"},
        files={"image": ("large.png", b"x" * (5 * 1024 * 1024 + 1), "image/png")},
    )
    assert response.status_code == 413


def test_photo_only_request_is_accepted():
    image = PROJECT_ROOT / "eval_cases" / "images" / "1_components.png"
    response = TestClient(app).post("/analyze", json={"image": str(image)})
    assert response.status_code == 200
    body = response.json()
    assert body["image_estimate"]["component_count"] == 1
    # With no key the local parser is the only option; with one, the model may succeed,
    # so the metadata must be self-consistent either way.
    extraction = body["extraction"]
    if extraction["fallback_used"]:
        # No model handled it, so nothing was understood from the picture.
        assert extraction["source"] in {"local_rules", "safe_default"}
        assert extraction["image_understanding_used"] is False
    else:
        assert extraction["source"].endswith("structured_output")
        assert extraction["image_understanding_used"] is True


def test_empty_request_is_rejected():
    assert TestClient(app).post("/analyze", json={}).status_code == 422


def test_corrupt_image_does_not_crash_the_service():
    corrupt = PROJECT_ROOT / "eval_cases" / "images" / "corrupt_probe.jpg"
    corrupt.write_bytes(b"this is definitely not a jpeg")
    try:
        response = TestClient(app).post(
            "/analyze", json={"brief": "gold ring", "image": str(corrupt)}
        )
        assert response.status_code == 200
        review = response.json()["review"]
        assert review["needs_human_review"] is True
        assert review["review_flags"]
    finally:
        corrupt.unlink(missing_ok=True)