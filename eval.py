"""Run the labelled evaluation set against the FastAPI application in-process.

Reports the metrics the brief asks for -- extraction field accuracy, top-1/top-3 hit
rate, count MAE and review-flag rate -- plus image-matching accuracy and the precision
and recall of the review flag on the cases designed to be ambiguous.

Run ``python eval.py`` for a deterministic offline summary. Use ``--live-models`` to
explicitly opt into configured provider calls, or ``--verbose`` for every case.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.matcher import load_inventory

PROJECT_ROOT = Path(__file__).resolve().parent
CASES_PATH = PROJECT_ROOT / "eval_cases" / "cases.json"

EXTRACTED_FIELDS = ("item_type", "metal", "purity", "gemstone", "target_weight_grams")


def _run_case(case: dict, client: TestClient, inventory: pd.DataFrame) -> dict:
    """Execute one labelled case and reduce the response to metric inputs."""
    payload: dict = {"brief": case.get("brief", "")}
    if case.get("image"):
        payload["image"] = str(PROJECT_ROOT / case["image"])

    response = client.post("/analyze", json=payload)
    response.raise_for_status()
    result = response.json()
    specs = result["specifications"]
    candidates = result["top_candidates"]

    extraction_accuracy = None
    expected_sku = case.get("expected_sku")
    if expected_sku:
        row = inventory.loc[expected_sku]
        matches = [
            specs["item_type"] == row["item_type"],
            specs["metal"] == row["metal"],
            specs["purity"] == row["purity"],
            specs["gemstone"] == row["gemstone"],
            abs(float(specs["target_weight_grams"]) - float(row["weight_grams"])) < 0.001,
        ]
        extraction_accuracy = sum(matches) / len(matches)

    count_error = None
    actual_count = (result["image_estimate"] or {}).get("component_count")
    expected_count = case.get("expected_components")
    if actual_count is not None and expected_count is not None:
        count_error = abs(actual_count - expected_count)

    predicted_sku = candidates[0]["sku"] if candidates else None
    item_type_hit = None
    if case.get("expected_item_type"):
        item_type_hit = int(candidates[0]["item_type"] == case["expected_item_type"])

    return {
        "case": case["id"],
        "kind": case.get("kind", "text"),
        "extraction_accuracy": extraction_accuracy,
        "top_1_hit": int(predicted_sku == expected_sku) if expected_sku else None,
        "top_3_hit": int(expected_sku in [c["sku"] for c in candidates]) if expected_sku else None,
        "image_top_1_hit": (
            int(predicted_sku == expected_sku) if case.get("kind") == "image" else None
        ),
        "retrieval_hit": (
            int(expected_sku in [r["sku"] for r in result["retrieval"].get("results", [])])
            if expected_sku
            else None
        ),
        "item_type_hit": item_type_hit,
        "count_error": count_error,
        "needs_review": int(result["review"]["needs_human_review"]),
        "expect_review": case.get("expect_review"),
        "image_used": int(bool(result["image_match"].get("used"))),
        "image_backend": result["image_match"].get("backend"),
        "confidence": result["review"]["confidence"],
        "predicted_sku": predicted_sku,
        "flags": "; ".join(result["review"]["review_flags"]),
    }


def _configure_offline_evaluation() -> tempfile.TemporaryDirectory:
    """Make the default evaluation repeatable and safe to run without API quota."""
    from app import catalog_index

    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        os.environ[name] = ""
    os.environ["LLM_PROVIDER"] = "offline"
    os.environ["IMAGE_EMBEDDING_BACKEND"] = "opencv"
    os.environ["INDEX_EMBEDDING_BACKEND"] = "opencv"
    os.environ["VISION_VERIFY_ENABLED"] = "0"
    # Keep evaluation independent of cached model descriptions, a running API process,
    # and any provider-specific index in the developer's data directory.
    temporary_store = tempfile.TemporaryDirectory(prefix="jewelry-estimate-eval-")
    catalog_index._close_clients()
    catalog_index.QDRANT_PATH = Path(temporary_store.name) / "qdrant"
    catalog_index.DESCRIPTION_CACHE = Path(temporary_store.name) / "descriptions.json"
    return temporary_store


def run_evaluation(live_models: bool = False) -> pd.DataFrame:
    """Execute all labelled cases and return one row of metrics per case.

    By default, all model calls are disabled and the local index is built as needed.
    Pass ``live_models=True`` only to explicitly opt into provider quota and latency.
    """
    temporary_store = None if live_models else _configure_offline_evaluation()
    try:
        cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        inventory = load_inventory().set_index("sku")
        client = TestClient(app)
        return pd.DataFrame([_run_case(case, client, inventory) for case in cases])
    finally:
        if temporary_store is not None:
            from app import catalog_index

            catalog_index._close_clients()
            temporary_store.cleanup()


def _rate(series: pd.Series) -> float | None:
    cleaned = series.dropna()
    return None if cleaned.empty else float(cleaned.mean())


def summarize(results: pd.DataFrame) -> dict:
    """Reduce per-case rows to the headline metrics."""
    image_rows = results[results["kind"] == "image"]
    image_only = results[results["kind"] == "image_only"]
    # On an image-only query whose catalog photo is shared across item types, the item
    # type is not knowable from pixels. Flagging such a request is the correct outcome,
    # so accuracy and "correctly refused" are reported together.
    image_only_flagged = (
        float(image_only["needs_review"].mean()) if not image_only.empty else None
    )
    labelled = results[results["expect_review"].notna()]
    should_flag = labelled[labelled["expect_review"] == 1]

    has_both_review_labels = (
        not labelled.empty and labelled["expect_review"].nunique(dropna=True) == 2
    )
    true_positives = int(
        (
            (labelled["expect_review"] == 1)
            & (labelled["needs_review"] == 1)
        ).sum()
    )
    predicted_positives_on_labelled = int(
        ((labelled["needs_review"] == 1)).sum()
    )
    precision = (
        true_positives / predicted_positives_on_labelled
        if has_both_review_labels and predicted_positives_on_labelled
        else None
    )
    recall = (
        float((should_flag["needs_review"] == 1).mean()) if not should_flag.empty else None
    )
    return {
        "cases": int(len(results)),
        "extraction_field_accuracy": _rate(results["extraction_accuracy"]),
        "top_1_hit_rate": _rate(results["top_1_hit"]),
        "top_3_hit_rate": _rate(results["top_3_hit"]),
        "image_top_1_hit_rate": _rate(image_rows["top_1_hit"]),
        "image_only_item_type_accuracy": _rate(results["item_type_hit"]),
        "image_only_flagged_for_review": image_only_flagged,
        "vector_recall_of_expected_sku": _rate(results["retrieval_hit"]),
        "count_mae": _rate(results["count_error"]),
        "review_flag_rate": float(results["needs_review"].mean()),
        "review_flag_precision_on_labelled": precision,
        "review_flag_recall_on_labelled": recall,
    }


def _print_table(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for name, value in metrics.items():
        if value is None:
            print(f"  {name:34s} n/a")
        elif name in {"cases", "count_mae"}:
            print(f"  {name:34s} {value:.2f}" if name == "count_mae" else f"  {name:34s} {value}")
        else:
            print(f"  {name:34s} {value:.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="print every case")
    parser.add_argument(
        "--live-models",
        action="store_true",
        help="use configured providers; may consume API quota and incur cost",
    )
    args = parser.parse_args()

    for noisy in (
        "app.extractor",
        "app.matcher",
        "app.embeddings",
        "app.vision_verify",
        "httpx",
        "httpx2",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    results = run_evaluation(live_models=args.live_models)
    backends = {b for b in results["image_backend"].dropna().tolist() if isinstance(b, str)}
    metrics = summarize(results)

    print(f"cases: {metrics['cases']} | image backend: {', '.join(sorted(backends)) or 'none'}")
    if args.verbose:
        print("\nPer-case results")
        print(
            results[
                [
                    "case",
                    "kind",
                    "predicted_sku",
                    "extraction_accuracy",
                    "top_1_hit",
                    "count_error",
                    "needs_review",
                    "confidence",
                ]
            ].to_string(index=False)
        )

    _print_table("Required metrics", {k: metrics[k] for k in (
        "extraction_field_accuracy",
        "top_1_hit_rate",
        "top_3_hit_rate",
        "count_mae",
        "review_flag_rate",
    )})
    _print_table("Image matching and review quality", {k: metrics[k] for k in (
        "image_top_1_hit_rate",
        "vector_recall_of_expected_sku",
        "image_only_item_type_accuracy",
        "image_only_flagged_for_review",
        "review_flag_precision_on_labelled",
        "review_flag_recall_on_labelled",
    )})
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())