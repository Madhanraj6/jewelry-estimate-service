"""Vector index over the jewelry catalog: images described as text, then embedded.

This is the retrieval half of the service. The catalog is ingested once:

1. Every ``data/images/<SKU>.jpg`` is passed to a vision model, which writes a plain
   text description ("polished yellow gold ring with a small red stone, photographed on
   a white background"). That description is cached next to the catalog so the VLM is
   paid for only when a photo changes.
2. Both the description text and the raw photo are embedded with the *same*
   multimodal model, so they land in one vector space. This is what lets a customer's
   photo and a customer's typed words query the same index.
3. Each SKU becomes one point with two named vectors (``description`` and ``image``) and
   a payload holding the full catalog row.

A customer's request is embedded the same way and searched, fusing the text and image
result lists with reciprocal rank fusion. Retrieval is semantic; the deterministic
weighted scorer in :mod:`app.matcher` still decides the final ranking, so a vector
database can widen the candidate set but can never silently choose a price.

Without ``GEMINI_API_KEY`` the whole pipeline still runs. Descriptions degrade to a
deterministic sentence built from the catalog attributes (clearly labelled as such),
text vectors use a hashing vectorizer, and image vectors use the OpenCV descriptor. The
index records which backend built it, and the response says so.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client.models import Distance, VectorParams

from .embeddings import (
    EmbeddingError,
    active_backend,
    embed_image,
    embedding_values,
    gemini_client,
    inventory_image_path,
    load_manifest,
    opencv_descriptor_dims,
    opencv_embed,
    read_image_bytes,
)
from .matcher import load_inventory

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "index"
QDRANT_PATH = Path(os.getenv("CATALOG_INDEX_PATH", str(INDEX_DIR / "qdrant"))).resolve()
DESCRIPTION_CACHE = Path(
    os.getenv("CATALOG_DESCRIPTION_CACHE", str(INDEX_DIR / "descriptions.json"))
).resolve()
COLLECTION = "catalog"

GEMINI_DESCRIPTION_MODEL = "gemini-3.8-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
OFFLINE_TEXT_DIMS = 256
RRF_K = 60  # Reciprocal rank fusion constant.

DESCRIPTION_PROMPT = (
    "Describe this jewelry photograph in one or two plain sentences for a catalog "
    "search index. State the product type, the metal colour, any visible stones and "
    "their colours, the setting style, and the background. Do not guess metal purity, "
    "carat weight, or price - only report what the image shows."
)


class IndexError_(RuntimeError):
    """Raised when the vector index cannot be built or queried."""


# --------------------------------------------------------------------- descriptions


def describe_image(
    sku: str, image_path: Path | None = None, allow_model: bool = True
) -> dict:
    """Return (and cache) the vision model's text description of a catalog photo.

    ``allow_model=False`` uses only what is already cached, falling back to the
    catalog attributes. This is how the index is rebuilt without spending generation
    quota on a catalog whose descriptions were written on an earlier run.
    """
    path = image_path or inventory_image_path(sku)
    if not path.is_file():
        return {"sku": sku, "description": "", "source": "missing_image", "model": None}

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = _load_descriptions()
    cached = cache.get(sku)
    have_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    fresh = bool(cached and cached.get("sha256") == digest and cached.get("description"))

    # A "quota_exhausted" entry is a placeholder, never a usable description.
    if fresh and cached.get("source") != "quota_exhausted":
        from_model = cached.get("source") == "vlm"
        if from_model or not have_key or not allow_model:
            return cached

    if not allow_model:
        return {
            "sku": sku,
            "description": attribute_description(sku),
            "source": "catalog_attributes",
            "model": None,
        }

    raw, mime = read_image_bytes(path)
    description = None
    model = None
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        try:
            description, model = _gemini_describe(raw, mime)
        except QuotaExhausted as exc:
            # Do not fall back silently: a quota stop is a "come back later" signal, not
            # a reason to write attribute descriptions and call the catalog finished.
            entry = {
                "sku": sku,
                "sha256": digest,
                "description": attribute_description(sku),
                "source": "quota_exhausted",
                "model": None,
                "note": str(exc),
            }
            cache[sku] = entry
            _save_descriptions(cache)
            raise
        except Exception as exc:  # Never let one bad photo stop the ingest.
            logger.warning("description failed for %s: %s", sku, exc)
    if not description:
        description, model = attribute_description(sku), "catalog_attributes"

    entry = {
        "sku": sku,
        "sha256": digest,
        "description": description,
        "source": "vlm" if model != "catalog_attributes" else "catalog_attributes",
        "model": model,
    }
    cache[sku] = entry
    _save_descriptions(cache)
    return entry


class QuotaExhausted(RuntimeError):
    """Raised when the provider refuses further generation requests for a period.

    On the free tier ``generate_content`` is capped per day, so the description pass has
    to stop cleanly and be resumable rather than retrying 36 times and burying the cause.
    """


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc).upper()
    return any(token in text for token in ("RESOURCE_EXHAUSTED", "429", "QUOTA"))


def _valid_description(text: str) -> bool:
    """Reject truncated or degenerate descriptions.

    A thinking model given a small token budget returns a description that stops
    mid-sentence ("This image displays four"). Indexing that is worse than indexing
    nothing, because it looks like a real observation to every downstream consumer.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) < 30:
        return False
    if not cleaned[-1] in ".!?\"')":
        return False
    # A description that is mostly one repeated word is a decode failure, not a caption.
    words = cleaned.lower().split()
    return len(set(words)) >= max(8, len(words) // 4)


def _gemini_describe(raw: bytes, mime: str) -> tuple[str, str]:
    from google.genai import types

    client = gemini_client()
    model = os.getenv("GEMINI_DESCRIPTION_MODEL", GEMINI_DESCRIPTION_MODEL)
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=DESCRIPTION_PROMPT),
                        types.Part.from_bytes(data=raw, mime_type=mime),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=400,
                # Gemini 3.x models think by default, and thinking tokens are billed
                # against max_output_tokens. With the default budget the visible answer
                # was being cut off mid-sentence, so thinking is switched off here.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    except Exception as exc:
        if _is_quota_error(exc):
            raise QuotaExhausted(str(exc)[:200]) from exc
        raise

    text = (response.text or "").strip()
    if not _valid_description(text):
        raise ValueError(f"description unusable (truncated or empty): {text[:80]!r}")
    return text, model


def attribute_description(sku: str) -> str:
    """Build a description from catalog attributes only.

    Used when no vision model is configured. It is explicitly *not* a description of the
    photograph - it restates the CSV row, which is why the entry is marked
    ``source="catalog_attributes"``.
    """
    inventory = load_inventory().set_index("sku")
    if sku not in inventory.index:
        return ""
    row = inventory.loc[sku]
    stone = "no visible stone" if str(row["gemstone"]) == "none" else f"a {row['gemstone']} stone"
    return (
        f"A {row['metal']} {row['item_type']} set in {row['purity']} with {stone}, "
        f"weighing {row['weight_grams']} grams."
    )


def _load_descriptions() -> dict:
    if DESCRIPTION_CACHE.is_file():
        try:
            return json.loads(DESCRIPTION_CACHE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("description cache is corrupt; rebuilding")
    return {}


def _save_descriptions(cache: dict) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    DESCRIPTION_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


# ------------------------------------------------------------------------ text side


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9.]+", str(text).lower())


def hash_text_vector(text: str, dims: int = OFFLINE_TEXT_DIMS) -> np.ndarray:
    """Deterministic hashing bag-of-words vector, used when no model is configured."""
    vector = np.zeros(dims, dtype=np.float32)
    for token in _tokenise(text):
        vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % dims] += 1.0
    norm = float(np.linalg.norm(vector))
    return (vector / norm).astype(np.float32) if norm else vector


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return (vector / norm).astype(np.float32) if norm else vector


def embed_text(text: str, backend: str, dims: int) -> np.ndarray:
    """Embed text with the active backend, reusing Gemini when it is configured."""
    if backend != "gemini":
        return hash_text_vector(text, dims)
    from google.genai import types

    client = gemini_client()
    model = os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL)
    response = client.models.embed_content(
        model=model,
        contents=[text],
        config=types.EmbedContentConfig(output_dimensionality=dims),
    )
    return _l2_normalize(embedding_values(response))


# ------------------------------------------------------------------------ the index


_OPEN_CLIENTS: list = []
_CLIENT = None


def _client():
    """Return the process-wide Qdrant client, creating it on first use.

    The local backend locks its storage directory, so exactly one client may exist per
    process. Opening a second one raises PermissionError, which is why this is memoized
    rather than constructed per call.
    """
    global _CLIENT
    if _CLIENT is None:
        from qdrant_client import QdrantClient

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        _CLIENT = QdrantClient(path=str(QDRANT_PATH))
    return _CLIENT


def _close_clients() -> None:
    """Release any live handle on the on-disk store.

    The local Qdrant backend locks its directory, so a rebuild has to drop the previous
    handle before the storage can be replaced.
    """
    global _CLIENT
    if _CLIENT is not None:
        try:
            _CLIENT.close()
        except Exception:  # Closing is best effort; a stale handle must not stop a rebuild.
            pass
        _CLIENT = None


def _vector_dims(backend: str) -> dict[str, int]:
    """Collection sizes for each named vector.

    Offline the text side is a 256-dim hashing vector while the image side is the
    OpenCV descriptor, so the two named vectors have different widths. Under Gemini both
    come from the same model and share one configurable width.
    """
    if backend != "gemini":
        return {"description": OFFLINE_TEXT_DIMS, "image": opencv_descriptor_dims()}
    dims = int(os.getenv("GEMINI_EMBEDDING_DIMS", "3072"))
    return {"description": dims, "image": dims}


def _image_vector(sku: str, backend: str, dims: int) -> np.ndarray | None:
    path = inventory_image_path(sku)
    if not path.is_file():
        return None
    try:
        if backend == "gemini":
            from google.genai import types

            raw, mime = read_image_bytes(path)
            client = gemini_client()
            response = client.models.embed_content(
                model=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL),
                contents=[types.Part.from_bytes(data=raw, mime_type=mime)],
                config=types.EmbedContentConfig(output_dimensionality=dims),
            )
            return _l2_normalize(embedding_values(response))
        return opencv_embed(read_image_bytes(path)[0])
    except EmbeddingError as exc:
        logger.warning("image vector failed for %s: %s", sku, exc)
        return None


def point_id(sku: str) -> int:
    """Deterministic unsigned point ID for a SKU.

    Must not use Python's ``hash()``: string hashing is salted per process, so every
    rebuild would assign fresh IDs and quietly double the collection instead of
    replacing its contents.
    """
    return int(hashlib.sha256(sku.encode("utf-8")).hexdigest()[:15], 16)


def indexable_text(sku: str, description: str) -> str:
    """Build the text that is actually embedded for a SKU.

    The vision description and the catalog attributes are both indexed, deliberately.
    A visual caption alone ("a yellow-metal band ring set with round, clear stones")
    matches a photo query well but fails a text query like "18K gold diamond ring
    weighing 3.2 grams", because it says none of those things. Attribute restatements
    alone have the opposite problem. Concatenating them lets one vector answer both.
    """
    inventory = load_inventory().set_index("sku")
    if sku not in inventory.index:
        return description
    row = inventory.loc[sku]
    stone = "no gemstone" if str(row["gemstone"]) == "none" else str(row["gemstone"])
    return (
        f"{description} "
        f"Catalog record: a {row['metal']} {row['item_type']} in {row['purity']} "
        f"with {stone}, weighing {row['weight_grams']} grams, priced at "
        f"{row['base_price']}."
    )


def build_index(force: bool = False, describe: bool = True) -> dict:
    """Describe and embed every catalog row, then upsert it into Qdrant.

    Safe to call repeatedly: descriptions and image embeddings are cached on disk and
    only recomputed when the underlying file changes. Pass ``describe=False`` to embed
    what is already cached without spending any generation quota.
    """
    from qdrant_client import models

    backend = os.getenv("INDEX_EMBEDDING_BACKEND") or active_backend()
    inventory = load_inventory()
    dims = _vector_dims(backend)

    if force:
        # Remove the store outright rather than calling delete_collection: the local
        # Qdrant backend keeps a lock on its directory and recreating a collection in
        # the same client can resurrect the previous segments.
        _close_clients()
        if QDRANT_PATH.exists():
            shutil.rmtree(QDRANT_PATH, ignore_errors=True)

    client = _client()

    exists = client.collection_exists(COLLECTION)
    if exists:
        current = client.get_collection(COLLECTION).config.params.vectors
        if isinstance(current, VectorParams):
            needs_rebuild = current.size != dims["description"]
        else:
            needs_rebuild = any(v.size != dims.get(name) for name, v in current.items())
        if needs_rebuild:
            logger.info("embedding dimensions changed; rebuilding the index")
            client.delete_collection(COLLECTION)
            exists = False
    if not exists:
        client.create_collection(
            COLLECTION,
            vectors_config={
                name: VectorParams(size=size, distance=Distance.COSINE)
                for name, size in dims.items()
            },
        )

    manifest = load_manifest()
    points, skipped = [], []
    for _, row in inventory.iterrows():
        sku = row["sku"]
        described = describe_image(sku, allow_model=describe)
        description = described["description"]
        if not description:
            skipped.append(sku)
            continue
        if described["source"] == "quota_exhausted":
            skipped.append(sku)
            continue
        try:
            description_vector = embed_text(
                indexable_text(sku, description), backend, dims["description"]
            )
        except Exception as exc:
            logger.error("description embedding failed for %s: %s", sku, exc)
            skipped.append(sku)
            continue
        image_vector = _image_vector(sku, backend, dims["image"])
        if image_vector is None:
            image_vector = np.zeros(dims["image"], dtype=np.float32)

        entry = manifest.get(sku, {})
        points.append(
            models.PointStruct(
                id=point_id(sku),
                vector={"description": description_vector.tolist(), "image": image_vector.tolist()},
                payload={
                    "sku": sku,
                    "item_type": row["item_type"],
                    "metal": row["metal"],
                    "purity": row["purity"],
                    "gemstone": row["gemstone"],
                    "weight_grams": float(row["weight_grams"]),
                    "base_price": float(row["base_price"]),
                    "image_url": row.get("image_url", ""),
                    "description": description,
                    "indexed_text": indexable_text(sku, description),
                    "description_source": described["source"],
                    "image_is_unique": bool(entry.get("image_is_unique", True)),
                    "has_image_vector": bool(image_vector.any()),
                },
            )
        )

    if points:
        client.upsert(COLLECTION, points)

    descriptions = _load_descriptions()
    vlm = sum(1 for e in descriptions.values() if e.get("source") == "vlm")
    quota_halted = sum(1 for e in descriptions.values() if e.get("source") == "quota_exhausted")
    attributes = sum(1 for e in descriptions.values() if e.get("source") == "catalog_attributes")
    mixed_sources = 0 < vlm < len(descriptions)
    if mixed_sources:
        logger.warning(
            "description sources are inconsistent: %d from the vision model, %d from catalog "
            "attributes. A partially described catalog embeds a few rows whose text has a "
            "completely different shape, which distorts retrieval. Re-run this script until "
            "every image has a vision-model description.",
            vlm,
            attributes,
        )
    return {
        "backend": backend,
        "dimensions": dims,
        "indexed": len(points),
        "skipped": skipped,
        "collection": COLLECTION,
        "path": str(QDRANT_PATH),
        "description_source": "vlm" if backend == "gemini" else "catalog_attributes",
        "descriptions_vlm": vlm,
        "descriptions_from_attributes": attributes,
        "descriptions_pending_quota": quota_halted,
        "description_sources_mixed": mixed_sources,
    }


def index_is_ready() -> bool:
    """Report whether a usable index exists without building one."""
    if not QDRANT_PATH.exists():
        return False
    try:
        client = _client()
        return client.collection_exists(COLLECTION) and client.count(COLLECTION).count > 0
    except Exception:
        return False


def index_backend_matches_active() -> tuple[bool, str | None]:
    """Check the stored collection still fits the configured embedding backend.

    Adding or removing an API key flips the backend from opencv to gemini, which changes
    the vector width. Querying anyway would fail deep inside the client with an opaque
    dimension error, so the mismatch is reported as an instruction to rebuild.
    """
    if not index_is_ready():
        return False, "vector index is not built"
    backend = os.getenv("INDEX_EMBEDDING_BACKEND") or active_backend()
    expected = _vector_dims(backend)
    stored = _client().get_collection(COLLECTION).config.params.vectors
    if isinstance(stored, VectorParams):
        stored = {"default": stored}
    for name, vector in stored.items():
        want = expected.get(name)
        if want is not None and vector.size != want:
            return False, (
                f"index was built with {vector.size}-d {name} vectors but the active "
                f"backend ({backend}) produces {want}-d; run "
                "`python scripts/build_catalog_index.py --force`"
            )
    return True, None


def _reciprocal_rank_fusion(rankings: list[list[str]], weights: list[float]) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for position, sku in enumerate(ranking):
            fused[sku] = fused.get(sku, 0.0) + weight / (RRF_K + position + 1)
    if not fused:
        return {}
    best = max(fused.values())
    return {sku: round(score / best, 4) for sku, score in fused.items()}


def _normalise_similarities(scores: list[float]) -> list[float]:
    """Min-max normalize raw cosine similarities into 0..1.

    Rank fusion is the wrong tool when there is only one ranking: it throws away the
    score margins and flattens everything to 1.000 / 0.984 / 0.968, which destroys the
    distinction the deterministic scorer needs. A single ranking therefore keeps its own
    similarity and is only rescaled, never re-ranked.
    """
    if not scores:
        return []
    low, high = min(scores), max(scores)
    span = high - low
    if span <= 1e-9:
        return [1.0] * len(scores)
    return [round((value - low) / span, 4) for value in scores]


def description_coverage() -> dict:
    """Report how much of the catalog has a real vision-model description.

    A partially described catalog is a genuine quality problem: the described rows are
    embedded as visual prose while the rest are embedded as attribute restatements, so
    the collection contains two unrelated text shapes and retrieval suffers. Callers
    surface this rather than presenting the index as fully described.
    """
    descriptions = _load_descriptions()
    total = len(descriptions)
    vlm = sum(1 for e in descriptions.values() if e.get("source") == "vlm")
    pending = sum(1 for e in descriptions.values() if e.get("source") == "quota_exhausted")
    return {
        "total": total,
        "described_by_vision_model": vlm,
        "from_catalog_attributes": total - vlm - pending,
        "pending_quota": pending,
        "complete": total > 0 and vlm == total,
        "mixed": 0 < vlm < total,
    }


def ensure_index() -> tuple[bool, str | None]:
    """Make sure the stored index is usable by the currently active backend.

    A reviewer who clones this without an API key runs the offline backend, whose
    vectors are a different width from the ones a Gemini-built index contains. Silently
    disabling vector search there would make the service look broken when the only fix
    is a rebuild that needs no key and takes a second. So: rebuild automatically when
    the rebuild is free, and explain clearly when it is not.
    """
    compatible, why = index_backend_matches_active()
    if compatible:
        return True, None

    backend = os.getenv("INDEX_EMBEDDING_BACKEND") or active_backend()
    if backend == "gemini":
        return False, why or "the index needs the Gemini backend"
    try:
        build_index(force=True, describe=False)
    except Exception as exc:
        return False, f"{why} (offline rebuild also failed: {exc})"
    return index_backend_matches_active()


def search(
    text: str | None = None,
    image: str | Path | bytes | None = None,
    limit: int = 5,
) -> dict:
    """Search the catalog semantically with whatever the caller supplied.

    Text and image hit separate named vectors and are fused, so a brief plus a photo uses
    both signals while a photo alone still works.
    """
    compatible, why = ensure_index()
    if not compatible:
        return {"used": False, "reason": why, "results": []}

    backend = os.getenv("INDEX_EMBEDDING_BACKEND") or active_backend()
    dims = _vector_dims(backend)
    client = _client()
    rankings: list[list[str]] = []
    similarity_lists: list[list[float]] = []
    weights: list[float] = []
    errors: list[str] = []

    if text and text.strip():
        try:
            vector = embed_text(text.strip(), backend, dims["description"])
            hits = client.query_points(
                COLLECTION, query=vector.tolist(), using="description", limit=limit
            ).points
            rankings.append([h.payload["sku"] for h in hits])
            similarity_lists.append(_normalise_similarities([h.score for h in hits]))
            weights.append(1.0)
        except Exception as exc:
            errors.append(f"text search failed: {type(exc).__name__}: {exc}")

    if image:
        try:
            vector, _ = embed_image(image, backend=backend)
            hits = client.query_points(
                COLLECTION, query=vector.tolist(), using="image", limit=limit
            ).points
            rankings.append([h.payload["sku"] for h in hits])
            similarity_lists.append(_normalise_similarities([h.score for h in hits]))
            weights.append(1.0)
        except Exception as exc:
            errors.append(f"image search failed: {type(exc).__name__}: {exc}")

    if not rankings:
        return {
            "used": False,
            "reason": "; ".join(errors) or "nothing to search with",
            "results": [],
        }

    if len(rankings) == 1:
        # One signal: keep its own similarity so the scorer can see the real margins.
        fused = dict(zip(rankings[0], similarity_lists[0]))
    else:
        fused = _reciprocal_rank_fusion(rankings, weights)

    payloads = {
        point.payload["sku"]: point.payload
        for point in client.scroll(COLLECTION, limit=10000, with_payload=True)[0]
        if point.payload and point.payload.get("sku")
    }
    results = []
    for sku, score in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]:
        payload = payloads.get(sku, {})
        results.append({"sku": sku, "semantic_score": score, **payload})
    return {
        "used": True,
        "backend": backend,
        "fusion": "similarity" if len(rankings) == 1 else "reciprocal_rank",
        "signals": len(rankings),
        "description_coverage": description_coverage(),
        "reason": "; ".join(errors) or None,
        "results": results,
    }


# ------------------------------------------------------------------ product details


def get_product(sku: str, include_image: bool = True) -> dict | None:
    """Return the full catalog row for a SKU, optionally with the image inlined.

    Never raises: this is called on the response path for every request, so a quota
    problem while writing a description must degrade to a row without one rather than
    turning a working catalog lookup into a 500.
    """
    inventory = load_inventory().set_index("sku")
    if sku not in inventory.index:
        return None
    row = inventory.loc[sku]
    try:
        described = describe_image(sku)
        description, description_source = described["description"], described["source"]
    except Exception as exc:
        logger.warning("description unavailable for %s: %s", sku, exc)
        description, description_source = attribute_description(sku), "catalog_attributes"
    image_metadata = load_manifest().get(sku, {})
    image_source = image_metadata.get("source")
    if image_source is None:
        if image_metadata.get("status") == "ok":
            image_source = (
                "inventory_url"
                if image_metadata.get("source_url") == image_metadata.get("catalog_url")
                else "fallback_url"
            )
        else:
            image_source = image_metadata.get("status", "unknown")
    image_metadata = load_manifest().get(sku, {})
    product = {
        "sku": sku,
        "item_type": row["item_type"],
        "metal": row["metal"],
        "purity": row["purity"],
        "gemstone": row["gemstone"],
        "weight_grams": float(row["weight_grams"]),
        "base_price": float(row["base_price"]),
        "image_url": row.get("image_url", ""),
        "image_path": str(inventory_image_path(sku).relative_to(PROJECT_ROOT).as_posix()),
        "image_source": image_source,
        "image_is_unique": image_metadata.get("image_is_unique"),
        "description": description,
        "description_source": description_source,
    }
    if include_image:
        path = inventory_image_path(sku)
        if path.is_file():
            raw = path.read_bytes()
            product["image_base64"] = base64.b64encode(raw).decode("ascii")
            product["image_mime"] = "image/jpeg"
    return product


def reset_index() -> None:
    """Delete the on-disk index so the next call rebuilds it from scratch."""
    _close_clients()
    if QDRANT_PATH.exists():
        shutil.rmtree(QDRANT_PATH, ignore_errors=True)
    if DESCRIPTION_CACHE.exists():
        DESCRIPTION_CACHE.unlink()