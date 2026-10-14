"""FastAPI entry point for the jewelry estimate service."""

import base64
import json
import mimetypes
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from .catalog_index import description_coverage, get_product, index_is_ready, search
from .config import api_key_configured
from .cv_estimator import estimate_components
from .embeddings import MAX_UPLOAD_BYTES, active_backend, inventory_image_path
from .extractor import extract_with_metadata
from .matcher import find_top_matches
from .review_engine import assess_review, field_confidence
from .schemas import EstimateRequest
from .vision_verify import verify_candidates

app = FastAPI(title="Jewelry Estimate Service", version="2.1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

#: Every catalog row is retrieved so that all of them receive a comparable semantic
#: score. A SKU missing from the result set would otherwise be scored as "maximally
#: dissimilar", which penalizes a perfect attribute match purely because retrieval
#: missed it. At this catalog size the full scan is free; past a few thousand rows this
#: becomes a real ANN search and the missing-row case needs an explicit "no semantic
#: evidence" state instead.
VECTOR_CANDIDATE_LIMIT = 64


def _service_warnings(
    retrieval: dict, image_metadata: dict, vision_verification: dict
) -> list[str]:
    """Catalog-wide conditions that degrade every request.

    These are deliberately kept out of ``review_flags``. A half-described index is true
    of the whole catalog, so folding it into per-request review would flag 100% of
    traffic and destroy the signal that review flags are supposed to carry. It is
    reported here instead: visible on every response and on /health.
    """
    warnings: list[str] = []
    coverage = (retrieval or {}).get("description_coverage") or {}
    if coverage.get("mixed"):
        warnings.append(
            f"only {coverage['described_by_vision_model']} of {coverage['total']} catalog "
            "photos have a vision-model description, so semantic retrieval compares visual "
            "prose against attribute restatements and its ranking is unreliable; re-run "
            "`python scripts/build_catalog_index.py` to finish it"
        )
    elif coverage and not coverage.get("complete"):
        warnings.append(
            "no catalog photo has a vision-model description yet; semantic retrieval is "
            "currently lexical over restated catalog rows"
        )
    if not retrieval.get("used") and retrieval.get("reason"):
        warnings.append(f"vector retrieval unavailable: {retrieval['reason']}")
    if (vision_verification or {}).get("reason"):
        warnings.append(f"vision verification skipped: {vision_verification['reason']}")
    return warnings


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    """Liveness plus the configuration a caller needs to debug "why is it offline?"."""
    return {
        "status": "ok",
        "vector_index_ready": index_is_ready(),
        "llm_key_configured": api_key_configured(),
        "image_backend": active_backend(),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        "gemini_fallback_model": os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite"),
        "embedding_model": os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
        "description_coverage": description_coverage(),
    }


@app.get("/inventory/{sku}")
def inventory_product(sku: str, include_image: bool = True) -> dict:
    """Full catalog row for one SKU, with the product photo inlined as base64."""
    product = get_product(sku.upper(), include_image=include_image)
    if product is None:
        raise HTTPException(status_code=404, detail=f"unknown SKU {sku}")
    return product


@app.get("/inventory/{sku}/image")
def inventory_image(sku: str) -> Response:
    """Serve the cached catalog photo for one SKU."""
    path = inventory_image_path(sku.upper())
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"no image for {sku}")
    return Response(content=path.read_bytes(), media_type="image/jpeg")


@app.post("/search")
def semantic_search(request: EstimateRequest) -> dict:
    """Pure vector-store retrieval over the catalog, without the deterministic re-rank."""
    return search(text=request.brief, image=request.image, limit=10)


@app.post(
    "/analyze",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "brief": {"type": "string"},
                            "image": {
                                "type": ["string", "null"],
                                "description": "Base64 image data URL or local path",
                            },
                        },
                        "anyOf": [
                            {"required": ["brief"]},
                            {"required": ["image"]},
                        ],
                    }
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "brief": {"type": "string"},
                            "image": {"type": "string", "format": "binary"},
                        },
                        "anyOf": [
                            {"required": ["brief"]},
                            {"required": ["image"]},
                        ],
                    }
                },
            },
        }
    },
)
async def analyze(request: Request) -> dict:
    """Extract, retrieve, match, optionally inspect an image, then decide review status."""
    estimate_request = await _parse_analyze_request(request)
    return _analyze(estimate_request)


async def _parse_analyze_request(request: Request) -> EstimateRequest:
    """Accept JSON data URLs/paths or multipart image uploads."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type == "application/json" or content_type.endswith("+json"):
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="Request body is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="JSON body must be an object")
        try:
            return EstimateRequest.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422, detail=exc.errors(include_context=False)
            ) from exc

    if content_type == "multipart/form-data":
        form = await request.form(max_files=1, max_fields=1)
        try:
            brief = form.get("brief", "")
            if not isinstance(brief, str):
                raise HTTPException(status_code=422, detail="brief must be a text field")

            image = form.get("image")
            if image is not None:
                if not isinstance(image, UploadFile):
                    raise HTTPException(status_code=422, detail="image must be an uploaded file")
                mime_type = image.content_type
                if not mime_type or mime_type == "application/octet-stream":
                    mime_type = mimetypes.guess_type(image.filename or "")[0]
                if mime_type:
                    mime_type = mime_type.split(";", 1)[0].strip().lower()
                if not mime_type or not mime_type.startswith("image/"):
                    raise HTTPException(status_code=415, detail="image must be an image file")
                if image.size is not None and image.size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Image must be 5 MB or smaller")
                raw = await image.read(MAX_UPLOAD_BYTES + 1)
                if len(raw) > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Image must be 5 MB or smaller")
                if not raw:
                    raise HTTPException(status_code=422, detail="image upload is empty")
                image_data = f"data:{mime_type};base64,{base64.b64encode(raw).decode('ascii')}"
            else:
                image_data = None

            try:
                return EstimateRequest.model_validate({"brief": brief, "image": image_data})
            except ValidationError as exc:
                raise HTTPException(
                    status_code=422, detail=exc.errors(include_context=False)
                ) from exc
        finally:
            await form.close()

    raise HTTPException(
        status_code=415,
        detail="Content-Type must be application/json or multipart/form-data",
    )


def _analyze(request: EstimateRequest) -> dict:
    """Run the estimation pipeline for a validated request."""
    specifications, extraction_metadata = extract_with_metadata(request.brief, request.image)

    retrieval = search(text=request.brief, image=request.image, limit=VECTOR_CANDIDATE_LIMIT)
    semantic_scores = {r["sku"]: r["semantic_score"] for r in retrieval.get("results", [])}

    candidates, image_metadata = find_top_matches(
        specifications,
        limit=3,
        query_image=request.image,
        semantic_scores=semantic_scores,
    )
    product_type_unresolved = bool(
        request.image and specifications.item_type.strip().lower() in {"", "unknown"}
    )
    image_metadata["product_type_unresolved"] = product_type_unresolved
    match = candidates[0]

    image_estimate = None
    image_error = None
    if request.image:
        try:
            image_estimate = estimate_components(request.image)
        except ValueError as exc:
            # An unavailable image must not silently create a high-confidence quote.
            image_error = str(exc)

    vision_verification = verify_candidates(request.brief, request.image, candidates)

    review = assess_review(
        match_score=match["match_score"],
        brief=request.brief,
        component_count=image_estimate["component_count"] if image_estimate else None,
        image_metadata=image_metadata,
        vision_verification=vision_verification,
        image_error=image_error,
        specifications=specifications.model_dump(),
    )
    for candidate in candidates:
        candidate["auto_selected"] = (
            candidate["sku"] == match["sku"] and review["auto_selected"]
        )
        candidate["recommendation_eligible"] = not product_type_unresolved

    warnings = _service_warnings(retrieval, image_metadata, vision_verification)

    confidence = field_confidence(
        specifications.model_dump(),
        extraction_metadata,
        match["match_score"],
        image_estimate,
        image_error,
        image_metadata=image_metadata,
        vision_verification=vision_verification,
        needs_human_review=review["needs_human_review"],
    )

    return {
        "specifications": specifications.model_dump(),
        "extraction": extraction_metadata,
        "match": match,
        "match_status": {
            "status": (
                "needs_product_type_confirmation"
                if product_type_unresolved
                else "candidate_available"
            ),
            "recommendation_available": not product_type_unresolved,
            "reason": (
                "The image did not provide a reliable jewelry type. Add a brief naming "
                "the item (for example, ring) or configure a multimodal model before "
                "using a catalog match."
                if product_type_unresolved
                else None
            ),
        },
        "product": get_product(match["sku"], include_image=True),
        "top_candidates": candidates,
        "retrieval": retrieval,
        "image_match": image_metadata,
        "vision_verification": vision_verification,
        "image_estimate": image_estimate,
        "review": review,
        "warnings": warnings,
        "field_confidence": confidence,
    }
