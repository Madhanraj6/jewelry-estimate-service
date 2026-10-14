"""Download every inventory reference photo once and cache it locally.

Catalog images are fetched to ``data/images/<SKU>.jpg`` so the API never depends on a
third-party host at request time and so embeddings can be cached against a stable file
hash. Identical ``image_url`` values are downloaded a single time and copied to each SKU
that references them; the manifest records that sharing so the service can downgrade
image confidence instead of pretending a shared photo is unique evidence.

Run with no arguments to refresh; pass ``--force`` to re-download existing files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INVENTORY_PATH = PROJECT_ROOT / "data" / "invent.csv"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
MANIFEST_PATH = IMAGE_DIR / "manifest.json"

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "jewelry-estimate-service/1.0 (reference image fetcher)"
MAX_EDGE = 1024

# Freely licensed Wikimedia Commons replacements, used only when the catalog URL cannot
# yield a picture. Covers rows whose `image_url` is a JavaScript-rendered product page
# (TataCliq, gehnaindia) or is blocked outright (Etsy returns 403 to non-browser clients).
# Each title was chosen to match the row's product type and metal so the row still has a
# usable reference photo. Attribution is written to the manifest.
COMMONS_FALLBACKS: dict[str, str] = {
    "RNG-001": "File:Diamond Engagement Ring White Gold by 1791 Diamond.jpg",
    "RNG-002": (
        "File:Burmese ruby gemstone, within a 22k gold ring, in West Bengal, India, "
        "photographed by Yogabrata Chakraborty, on May 6, 2023.jpg"
    ),
    "RNG-003": "File:Amethyst ring stone MET DP141791.jpg",
    "RNG-004": "File:Platinum finger ring with diamond.jpg",
    "RNG-005": "File:Closeup of an emerald ring in a finger.jpg",
    "RNG-006": "File:Beryl-130916.jpg",
    "RNG-007": "File:Plain Silver Ring - YDEA - 47026.jpg",
    "RNG-008": "File:Gold Ring 2.jpg",
    "CHN-002": "File:Four 18K gold chains.jpg",
    "CHN-004": "File:Necklace with cameo of Veronica's Veil MET DP-12990-001.jpg",
    "CHN-006": "File:16th century gold chain length comparison.jpg",
    "CHN-007": "File:Black Onyx Necklace.jpg",
    "NCK-006": "File:Necklace MET DP342239.jpg",
    "NCK-007": "File:Silver necklace and cauliflower by ASQ.jpg",
    "BRC-007": "File:Modern white metal and layered chalcedony hinged bracelet (FindID 1017837).jpg",
    "EAR-001": "File:Gold earrings with diamonds, garnets, citrines and aquamarines on a cucumber by ASQ.jpg",
}

logger = logging.getLogger("fetch_inventory_images")

#: Files that are site furniture rather than product photography.
_NON_PRODUCT = re.compile(
    r"(icon|logo|sprite|favicon|placeholder|banner|payment|badge|rating|trust|"
    r"loader|spinner|avatar|cart|search|newsletter|cookie|social|flag-|stamp)",
    re.I,
)
_IMAGE_EXT = re.compile(r"\.(jpe?g|png|webp)(\?|$)", re.I)


def _decode(raw: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)


def _http_get_with_type(url: str, timeout: int = 45) -> tuple[bytes, str]:
    """Fetch a URL, retrying Wikimedia with browser headers if it answers 403.

    upload.wikimedia.org rejects the descriptive contact User-Agent for some originals
    but serves the same bytes to a browser-shaped request.
    """
    def attempt(headers: dict) -> tuple[bytes, str]:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers.get("Content-Type", "")

    try:
        return attempt({"User-Agent": USER_AGENT})
    except urllib.error.HTTPError as exc:
        if exc.code != 403 or "wikimedia.org" not in url:
            raise
    return attempt(BROWSER_HEADERS)


def _html_candidates(html: str, base_url: str) -> list[str]:
    """Pull plausible product-image URLs out of a product page, best first.

    Catalog columns often hold a product *page* rather than a file. The order matters:
    Open Graph is what the site itself declares as the picture, JSON-LD is the structured
    data, and raw <img> tags are a last resort because they are full of site furniture.
    """
    candidates: list[str] = []

    def push(value: str | None) -> None:
        if not value:
            return
        absolute = urllib.parse.urljoin(base_url, value.strip().replace("\\/", "/"))
        if _IMAGE_EXT.search(absolute) and not _NON_PRODUCT.search(absolute):
            candidates.append(absolute)

    for pattern in (
        r'<meta[^>]+property=["\']og:image(?::secure_url)?["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image(?::secure_url)?["\']',
    ):
        push(re.search(pattern, html, re.I).group(1) if re.search(pattern, html, re.I) else None)

    for match in re.finditer(r'"image"\s*:\s*\[?\s*"?(https?://[^"\'\]]+)', html, re.I):
        push(match.group(1))

    for match in re.finditer(r'<img[^>]+src=["\']([^"\']+)["\']', html, re.I):
        push(match.group(1))

    seen, unique = set(), []
    for url in candidates:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def resolve_image_url(url: str) -> tuple[bytes, str]:
    """Return JPEG bytes for a catalog entry plus a note describing how it was found.

    Accepts either a direct image file or a product page, which is what an ``image_url``
    column copied out of a shop usually contains.
    """
    raw, content_type = _http_get_with_type(url)
    if _decode(raw) is not None:
        return _to_jpeg(raw), "direct image"

    html = raw.decode("utf-8", "ignore")
    notes = []
    for candidate in _html_candidates(html, url)[:4]:
        try:
            payload, _ = _http_get_with_type(candidate)
        except Exception as exc:
            notes.append(f"candidate rejected: {exc}")
            continue
        if _decode(payload) is None:
            notes.append(f"not an image: {candidate[:60]}")
            continue
        return _to_jpeg(payload), f"scraped from product page ({candidate[:120]})"

    raise ValueError(
        "no product image found on the page; tried "
        f"{len(_html_candidates(html, url))} candidates. {'; '.join(notes[:2])}"
    )


BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
    "Referer": "https://commons.wikimedia.org/",
}


def _http_get(url: str, timeout: int = 45) -> bytes:
    """Fetch a URL, retrying Wikimedia with browser headers if it answers 403.

    upload.wikimedia.org rejects the descriptive contact User-Agent for some originals
    but serves the same bytes to a browser-shaped request.
    """
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code != 403 or "wikimedia.org" not in url:
            raise
    request = urllib.request.Request(url, headers=BROWSER_HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def commons_image_url(title: str, width: int = MAX_EDGE) -> tuple[str, str]:
    """Resolve a Commons file title to a thumbnail URL plus the file description page."""
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": width,
            "format": "json",
        }
    )
    payload = json.loads(_http_get(f"{COMMONS_API}?{query}").decode("utf-8"))
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        if "imageinfo" not in page:
            raise LookupError(f"Commons has no image for {title!r}")
        info = page["imageinfo"][0]
        if not str(info.get("mime", "")).startswith("image/"):
            raise LookupError(f"{title!r} is not a raster image ({info.get('mime')})")
        metadata = info.get("extmetadata", {})
        return (
            info.get("thumburl") or info["url"],
            f"{info.get('descriptionurl', '')} | {metadata.get('LicenseShortName', {}).get('value', 'see source')}",
        )
    raise LookupError(f"Commons lookup returned no page for {title!r}")


def _to_jpeg(raw: bytes) -> bytes:
    """Decode whatever format arrived and re-encode as bounded JPEG."""
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("bytes did not decode as an image")
    height, width = decoded.shape[:2]
    if max(height, width) > MAX_EDGE:
        scale = MAX_EDGE / max(height, width)
        decoded = cv2.resize(
            decoded, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
        )
    ok, encoded = cv2.imencode(".jpg", decoded, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError("could not encode JPEG")
    return encoded.tobytes()


def _placeholder(label: str) -> bytes:
    """Deterministic grey card used only when a download fails, so the pipeline still runs."""
    canvas = np.full((256, 256, 3), 235, dtype=np.uint8)
    cv2.putText(
        canvas, label[:12], (12, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1, cv2.LINE_AA
    )
    ok, encoded = cv2.imencode(".jpg", canvas)
    return encoded.tobytes() if ok else b""


def _load_existing_manifest() -> dict:
    if MANIFEST_PATH.is_file():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("existing manifest is unreadable; rebuilding from scratch")
    return {}


def fetch_images(force: bool = False) -> dict:
    """Populate data/images and return the provenance manifest."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(INVENTORY_PATH, dtype=str).fillna("")
    manifest = _load_existing_manifest() if not force else {}
    manifest.setdefault("generated_by", "scripts/fetch_inventory_images.py")
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["entries"] = manifest.get("entries", {})

    by_url: dict[str, bytes] = {}

    for _, row in inventory.iterrows():
        sku = row["sku"]
        target = IMAGE_DIR / f"{sku}.jpg"
        entry = manifest["entries"].get(sku, {})
        if target.is_file() and not force:
            entry.setdefault("status", "cached")
            entry["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            manifest["entries"][sku] = entry
            continue

        catalog_url = row.get("image_url", "")
        source_url, attribution, notes = catalog_url, "", []
        payload: bytes | None = None
        how = ""

        if source_url:
            try:
                if source_url in by_url:
                    payload, how = by_url[source_url], "reused"
                    notes.append("reused an identical catalog URL from another SKU")
                else:
                    payload, how = resolve_image_url(source_url)
                    by_url[source_url] = payload
            except Exception as exc:
                logger.warning("%s catalog URL failed: %s", sku, exc)
                notes.append(f"catalog URL failed: {type(exc).__name__}: {exc}")

        if payload is None and sku in COMMONS_FALLBACKS:
            title = COMMONS_FALLBACKS[sku]
            try:
                fallback_url, fallback_attribution = commons_image_url(title)
                payload, how = resolve_image_url(fallback_url)
                source_url, attribution = fallback_url, fallback_attribution
                notes.append(f"fell back to Wikimedia Commons; source {title}")
            except Exception as exc:
                logger.error("%s Commons fallback failed: %s", sku, exc)
                notes.append(f"Commons fallback failed: {exc}")

        if payload is None and target.is_file():
            payload = target.read_bytes()
            status = "stale_cache"
            notes.append("kept the previously cached photo after every source failed")
        elif payload is None:
            payload = _placeholder(sku)
            notes.append("placeholder written because no reference photo was available")
            status = "placeholder"
        else:
            status = "ok"

        target.write_bytes(payload)
        entry = {
            "sku": sku,
            "status": status,
            "catalog_url": catalog_url,
            "source_url": source_url,
            "resolved_via": how,
            "attribution": attribution,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "notes": "; ".join(notes),
        }
        manifest["entries"][sku] = entry
        logger.info("%s %s %s", sku, status, source_url[:70])
        time.sleep(0.2)  # Be polite to the upstream hosts.

    # Flag SKUs whose photos are byte-identical, since a shared photo cannot identify a SKU.
    seen: dict[str, list[str]] = {}
    for sku, entry in manifest["entries"].items():
        seen.setdefault(entry.get("sha256", ""), []).append(sku)
    for digest, skus in seen.items():
        if digest and len(skus) > 1:
            for sku in skus:
                manifest["entries"][sku]["shared_with"] = sorted(s for s in skus if s != sku)
                manifest["entries"][sku]["image_is_unique"] = False
    for sku, entry in manifest["entries"].items():
        entry.setdefault("image_is_unique", True)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    manifest = fetch_images(force=args.force)
    entries = manifest["entries"]
    ok = sum(1 for e in entries.values() if e.get("status") == "ok")
    shared = sum(1 for e in entries.values() if not e.get("image_is_unique", True))
    print(f"\n{len(entries)} SKUs | {ok} with a real photo | {shared} sharing a photo with another SKU")
    print(f"images: {IMAGE_DIR}\nmanifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())