"""Image embeddings for catalog matching, with an offline fallback.

Two interchangeable backends produce vectors for the same catalog photo:

``gemini``
    ``gemini-embedding-2``, the Gemini API's multimodal embedding model. It places
    photos and text in one space, so a customer's snapshot can be compared directly
    against catalog photographs. Requires ``GEMINI_API_KEY``.

``opencv``
    A hand-rolled colour/shape descriptor built from HSV histograms, Lab moments and
    edge orientation. It needs no key and no downloaded weights, and it is genuinely
    useful for this catalog because metal colour (gold vs silver vs rose gold vs
    platinum) dominates its appearance.

Every vector is cached on disk under ``data/.cache/embeddings`` keyed by the backend
name and the SHA-256 of the source bytes, so repeated requests are free and evaluation
runs are reproducible regardless of which backend is active.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / ".cache" / "embeddings"
INVENTORY_IMAGE_DIR = PROJECT_ROOT / "data" / "images"
MANIFEST_PATH = INVENTORY_IMAGE_DIR / "manifest.json"

GEMINI_MODEL = "gemini-embedding-2"
DESCRIPTOR_SIZE = 224
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class EmbeddingError(RuntimeError):
    """Raised when an image cannot be embedded by the active backend."""


# --------------------------------------------------------------------------- loading


def inventory_image_path(sku: str) -> Path:
    """Return the cached reference photo for a SKU."""
    return INVENTORY_IMAGE_DIR / f"{sku}.jpg"


def load_manifest() -> dict:
    """Return the image provenance manifest, or an empty mapping when absent."""
    if not MANIFEST_PATH.is_file():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get("entries", {})
    except json.JSONDecodeError:
        logger.warning("image manifest is not valid JSON; treating images as unverified")
        return {}


def image_is_unique(sku: str) -> bool:
    """Report whether this SKU's photo is not byte-identical to another SKU's."""
    entry = load_manifest().get(sku)
    return bool(entry) and bool(entry.get("image_is_unique", True))


def read_image_bytes(image_input: str | Path | bytes) -> tuple[bytes, str]:
    """Normalize a path, data URL, or raw upload into (bytes, mime type)."""
    if isinstance(image_input, (bytes, bytearray)):
        return bytes(image_input), "image/jpeg"

    value = str(image_input)
    if value.startswith("data:image/"):
        import base64

        header, _, encoded = value.partition(",")
        mime = header.split(";")[0].removeprefix("data:")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise EmbeddingError("image data URL is not valid base64") from exc
        if len(raw) > MAX_UPLOAD_BYTES:
            raise EmbeddingError("image must be 5 MB or smaller")
        return raw, mime

    path = Path(value)
    if not path.is_file():
        raise EmbeddingError(f"image not found: {value}")
    raw = path.read_bytes()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise EmbeddingError("image must be 5 MB or smaller")
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return raw, mime


def _decode(raw: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise EmbeddingError("image bytes could not be decoded")
    return image


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise EmbeddingError("embedding produced a zero vector")
    return (vector / norm).astype(np.float32)


# ------------------------------------------------------------------ offline descriptor


def _histogram(channel: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    counts, _ = np.histogram(channel, bins=bins, range=value_range, density=False)
    return counts.astype(np.float32)


def _colour_descriptor(image: np.ndarray) -> np.ndarray:
    """HSV histograms plus Lab channel statistics describing metal and stone colour."""
    resized = cv2.resize(image, (DESCRIPTOR_SIZE, DESCRIPTOR_SIZE), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)

    features: list[np.ndarray] = [
        _histogram(hsv[:, :, 0], 24, (0, 180)),  # hue: gold vs silver vs rose gold
        _histogram(hsv[:, :, 1], 8, (0, 256)),  # saturation
        _histogram(hsv[:, :, 2], 8, (0, 256)),  # brightness
    ]
    for channel in (lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]):
        features.append(np.array([channel.mean(), channel.std()], dtype=np.float32))
    return np.concatenate(features)


def _shape_descriptor(image: np.ndarray) -> np.ndarray:
    """Edge orientation, edge density and contour geometry."""
    resized = cv2.resize(image, (DESCRIPTOR_SIZE, DESCRIPTOR_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 60, 160)
    orientation = np.zeros(8, dtype=np.float32)
    angles = cv2.cartToPolar(
        cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3),
        angleInDegrees=True,
    )[1]
    valid = edges > 0
    if valid.any():
        weights = edges[valid].astype(np.float32)
        np.add.at(orientation, np.minimum((angles[valid] // 45).astype(int), 7), weights)
    orientation = orientation / max(float(orientation.sum()), 1.0)
    edge_density = np.array([float(edges.mean()) / 255.0], dtype=np.float32)

    _, threshold = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    geometry = np.zeros(4, dtype=np.float32)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        perimeter = float(cv2.arcLength(largest, True))
        _, (width, height), _ = cv2.minAreaRect(largest)
        geometry = np.array(
            [
                area / (DESCRIPTOR_SIZE * DESCRIPTOR_SIZE),
                4.0 * np.pi * area / (perimeter * perimeter) if perimeter else 0.0,  # circularity
                width / DESCRIPTOR_SIZE,
                height / DESCRIPTOR_SIZE,
            ],
            dtype=np.float32,
        )
    return np.concatenate([orientation, edge_density, geometry])


def opencv_embed(raw: bytes) -> np.ndarray:
    """Deterministic, dependency-light descriptor for a catalog or customer photo."""
    image = _decode(raw)
    return _l2_normalize(np.concatenate([_colour_descriptor(image), _shape_descriptor(image)]))


def opencv_descriptor_dims() -> int:
    """Length of the OpenCV descriptor, needed to size a collection before indexing."""
    blank = np.zeros((DESCRIPTOR_SIZE, DESCRIPTOR_SIZE, 3), dtype=np.uint8)
    return int(
        np.concatenate([_colour_descriptor(blank), _shape_descriptor(blank)]).shape[0]
    )


# ------------------------------------------------------------------------ gemini path


def _gemini_client():
    """Deprecated alias kept so existing call sites keep working."""
    return gemini_client()


_GEMINI_CLIENT = None


def gemini_client():
    """Return a process-wide Gemini client.

    Constructing a client per call is wasteful and actively harmful: the SDK's underlying
    transport is shared, so a client that gets garbage collected can close the pool out
    from under an in-flight request, surfacing as "Cannot send a request, as the client
    has been closed". One client per process avoids it and reuses connections.
    """
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EmbeddingError("GEMINI_API_KEY is not configured")
        _GEMINI_CLIENT = genai.Client(api_key=api_key)
    return _GEMINI_CLIENT


def reset_gemini_client() -> None:
    """Drop the cached client, e.g. after the key changes."""
    global _GEMINI_CLIENT
    _GEMINI_CLIENT = None


def embedding_values(response) -> np.ndarray:
    """Pull the vector out of an ``embed_content`` response.

    The response shape is not stable across Gemini models: some return a single
    ``ContentEmbedding`` with a ``.values`` attribute, while ``gemini-embedding-2``
    returns a *list* of them. Both are accepted here so a model swap cannot silently
    turn every embedding into a "no values" failure.
    """
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None:
        raise EmbeddingError("Gemini response contained no embeddings field")
    if isinstance(embeddings, list):
        if not embeddings:
            raise EmbeddingError("Gemini returned an empty embeddings list")
        embeddings = embeddings[0]
    values = getattr(embeddings, "values", None)
    if values is None:
        raise EmbeddingError("Gemini embedding had no values attribute")
    return np.asarray(values, dtype=np.float32)


def gemini_embed(raw: bytes, mime: str) -> np.ndarray:
    """Embed one image with the Gemini multimodal embedding model."""
    from google.genai import types

    model = os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_MODEL)
    part = types.Part.from_bytes(data=raw, mime_type=mime)
    response = _gemini_client().models.embed_content(model=model, contents=[part])
    return _l2_normalize(embedding_values(response))


# ------------------------------------------------------------------------ public facade


def active_backend() -> str:
    """Return the backend that will actually be used for new embeddings."""
    override = os.getenv("IMAGE_EMBEDDING_BACKEND", "").strip().lower()
    if override in {"gemini", "opencv"}:
        return override
    return "gemini" if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) else "opencv"


def _cache_path(backend: str, digest: str) -> Path:
    return CACHE_DIR / backend / f"{digest}.npy"


def embed_image(image_input: str | Path | bytes, backend: str | None = None) -> tuple[np.ndarray, dict]:
    """Embed an image, returning the unit vector plus metadata about how it was produced."""
    chosen = (backend or active_backend()).lower()
    raw, mime = read_image_bytes(image_input)
    digest = _sha256(raw)
    cache_path = _cache_path(chosen, digest)

    if cache_path.is_file():
        try:
            return np.load(cache_path), {
                "backend": chosen,
                "cached": True,
                "sha256": digest,
                "dimensions": int(np.load(cache_path).shape[0]),
            }
        except (OSError, ValueError):
            logger.warning("embedding cache entry %s is unreadable; recomputing", cache_path)

    if chosen == "gemini":
        try:
            vector = gemini_embed(raw, mime)
        except EmbeddingError:
            raise
        except Exception as exc:
            # Provider errors (bad image bytes, quota, overload) are all "cannot embed
            # this image", and callers handle one failure type.
            raise EmbeddingError(f"{type(exc).__name__}: {exc}"[:200]) from exc
    elif chosen == "opencv":
        vector = opencv_embed(raw)
    else:
        raise EmbeddingError(f"unknown embedding backend {chosen!r}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.save(cache_path, vector)
    except OSError as exc:
        logger.warning("could not write embedding cache entry %s: %s", cache_path, exc)
    return vector, {
        "backend": chosen,
        "cached": False,
        "sha256": digest,
        "dimensions": int(vector.shape[0]),
    }


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity of two unit vectors, clamped to [-1, 1]."""
    return float(np.clip(np.dot(left, right), -1.0, 1.0))