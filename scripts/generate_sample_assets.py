"""Generate original synthetic jewelry images and an MP4 demo for the project.

Run this in the project root with:
    python scripts/generate_sample_assets.py

The generated images are illustrations for demos and offline testing, not product photos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "data" / "invent.csv"
IMAGE_DIR = ROOT / "data" / "images"
MEDIA_DIR = ROOT / "examples" / "media"
SIZE = 640

METALS = {
    "gold": (34, 177, 243),
    "rose gold": (100, 155, 220),
    "silver": (190, 198, 205),
    "platinum": (174, 184, 192),
}
GEMS = {
    "diamond": (235, 245, 250),
    "ruby": (45, 55, 205),
    "emerald": (55, 150, 70),
    "sapphire": (190, 80, 25),
    "amethyst": (175, 80, 165),
    "pearl": (225, 230, 235),
    "turquoise": (190, 180, 45),
    "onyx": (40, 40, 45),
    "morganite": (145, 175, 225),
    "none": (220, 225, 230),
}


def _gemstone_color(name: str) -> tuple[int, int, int]:
    name = str(name).lower()
    return next((color for gem, color in GEMS.items() if gem in name), GEMS["diamond"])


def draw_jewelry(item_type: str, metal: str, gemstone: str, seed: int = 0) -> np.ndarray:
    """Render a simple, original illustration of one jewelry category."""
    image = np.full((SIZE, SIZE, 3), 250, dtype=np.uint8)
    metal_color = METALS.get(str(metal).lower(), METALS["gold"])
    gem_color = _gemstone_color(gemstone)
    center = (SIZE // 2, SIZE // 2)
    item_type = str(item_type).lower()

    if item_type == "ring":
        cv2.ellipse(image, center, (170, 135), -12, 0, 360, metal_color, 31, cv2.LINE_AA)
        cv2.ellipse(image, (center[0], center[1] - 4), (128, 94), -12, 0, 360, (250, 250, 250), -1, cv2.LINE_AA)
        for offset in (-52, -26, 0, 26, 52):
            cv2.circle(image, (center[0] + offset, center[1] - 120), 13 if offset == 0 else 8, gem_color, -1, cv2.LINE_AA)
            cv2.circle(image, (center[0] + offset, center[1] - 120), 13 if offset == 0 else 8, (160, 145, 125), 1, cv2.LINE_AA)
    elif item_type == "bracelet":
        cv2.ellipse(image, center, (210, 132), 0, 0, 360, metal_color, 34, cv2.LINE_AA)
        cv2.ellipse(image, center, (166, 91), 0, 0, 360, (250, 250, 250), -1, cv2.LINE_AA)
        for angle in range(210, 331, 30):
            point = (center[0] + int(190 * np.cos(np.deg2rad(angle))), center[1] + int(115 * np.sin(np.deg2rad(angle))))
            cv2.circle(image, point, 9, gem_color, -1, cv2.LINE_AA)
    elif item_type in {"chain", "necklace"}:
        points = []
        for i in range(41):
            t = i / 40
            x = int(112 + 416 * t)
            y = int(180 + 250 * (abs(2 * t - 1) ** 1.5))
            points.append((x, y))
        cv2.polylines(image, [np.array(points, dtype=np.int32)], False, metal_color, 8, cv2.LINE_AA)
        for i, point in enumerate(points[::2]):
            cv2.ellipse(image, point, (8, 5), (i % 2) * 40, 0, 360, (155, 135, 110), 2, cv2.LINE_AA)
        if item_type == "necklace":
            cv2.circle(image, points[-1], 39, metal_color, 8, cv2.LINE_AA)
            cv2.circle(image, points[-1], 23, gem_color, -1, cv2.LINE_AA)
    elif item_type == "pendant":
        cv2.ellipse(image, (center[0], 135), (22, 34), 0, 0, 360, metal_color, 10, cv2.LINE_AA)
        cv2.line(image, (center[0], 168), (center[0], 205), metal_color, 8, cv2.LINE_AA)
        cv2.circle(image, center, 105, metal_color, 20, cv2.LINE_AA)
        cv2.circle(image, center, 75, gem_color, -1, cv2.LINE_AA)
        cv2.circle(image, center, 39, (240, 240, 240), -1, cv2.LINE_AA)
        cv2.circle(image, center, 22, gem_color, -1, cv2.LINE_AA)
        cv2.circle(image, (center[0], 463), 27, gem_color, -1, cv2.LINE_AA)
        cv2.line(image, (center[0], 394), (center[0], 434), metal_color, 8, cv2.LINE_AA)
    elif item_type in {"earring", "earrings"}:
        for x in (center[0] - 112, center[0] + 112):
            cv2.ellipse(image, (x, center[1] - 28), (56, 76), 0, 0, 360, metal_color, 13, cv2.LINE_AA)
            cv2.circle(image, (x, center[1] + 65), 27, gem_color, -1, cv2.LINE_AA)
            cv2.circle(image, (x, center[1] + 65), 27, metal_color, 4, cv2.LINE_AA)
    else:
        cv2.circle(image, center, 118, metal_color, 20, cv2.LINE_AA)
        cv2.circle(image, center, 74, gem_color, -1, cv2.LINE_AA)

    rng = np.random.default_rng(seed)
    caption = f"{item_type.title()} illustration"
    cv2.putText(image, caption, (22, SIZE - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (110, 115, 125), 1, cv2.LINE_AA)
    return image


def _rounded_card(frame: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int], fill=(255, 255, 255)) -> None:
    cv2.rectangle(frame, top_left, bottom_right, fill, -1, cv2.LINE_AA)
    cv2.rectangle(frame, top_left, bottom_right, (225, 229, 235), 2, cv2.LINE_AA)


def _demo_frame(reference: np.ndarray, step: int) -> np.ndarray:
    width, height = 1280, 720
    frame = np.full((height, width, 3), (247, 248, 251), dtype=np.uint8)
    cv2.putText(frame, "JEWELRY ESTIMATE SERVICE", (54, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (70, 82, 98), 2, cv2.LINE_AA)
    captions = [
        ("1. Add a reference", "Upload a product image and a short brief."),
        ("2. Extract supported specs", "Vision models help identify visible product details."),
        ("3. Match, then review", "A candidate is not a confirmed quote."),
    ]
    title, subtitle = captions[step]
    cv2.putText(frame, title, (54, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (34, 50, 70), 3, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (54, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 110, 125), 1, cv2.LINE_AA)
    _rounded_card(frame, (54, 235), (584, 646))
    _rounded_card(frame, (626, 235), (1226, 646))
    preview = cv2.resize(reference, (450, 450), interpolation=cv2.INTER_AREA)
    frame[270:720, 94:544] = preview
    cv2.putText(frame, "SYNTHETIC REFERENCE", (98, 616), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 110, 125), 1, cv2.LINE_AA)
    details = [
        (["Waiting for analysis", "No catalog guess is shown yet."], (110, 120, 132)),
        (["Item type   ring", "Metal        gold", "Purity       18K", "Gemstone     diamond"], (45, 115, 75)),
        (["Closest candidate: RNG-001", "Similarity: illustrative only", "Status: human review", "Catalog price is not a quote."], (45, 80, 145)),
    ]
    lines, color = details[step]
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (678, 340 + index * 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    for index in range(3):
        cv2.circle(frame, (54 + index * 40, 684), 8, (183, 121, 31) if index == step else (215, 220, 228), -1, cv2.LINE_AA)
    return frame


def create_demo_video(reference: np.ndarray, output: Path) -> None:
    width, height, fps = 1280, 720, 15
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize an MP4 encoder")
    try:
        for step in range(3):
            frame = _demo_frame(reference, step)
            for _ in range(30):
                writer.write(frame)
    finally:
        writer.release()


def _gif_palette() -> bytes:
    palette = bytearray()
    for red in range(8):
        for green in range(8):
            for blue in range(4):
                palette.extend((round(red * 255 / 7), round(green * 255 / 7), round(blue * 255 / 3)))
    return bytes(palette)


def _gif_image_data(frame: np.ndarray) -> bytes:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pixels = (
        ((rgb[:, :, 0] >> 5) << 5)
        | ((rgb[:, :, 1] >> 5) << 2)
        | (rgb[:, :, 2] >> 6)
    ).astype(np.uint8).ravel()

    # Clearing after every pixel keeps the GIF LZW code width fixed and makes the
    # small, hand-encoded demo asset straightforward to regenerate without dependencies.
    codes = np.empty(pixels.size * 2 + 2, dtype=np.uint16)
    codes[0] = 256
    codes[1:-1:2] = pixels
    codes[2:-1:2] = 256
    codes[-1] = 257
    packed = bytearray()
    bit_buffer = 0
    bit_count = 0
    for code in codes:
        bit_buffer |= int(code) << bit_count
        bit_count += 9
        while bit_count >= 8:
            packed.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bit_count -= 8
    if bit_count:
        packed.append(bit_buffer & 0xFF)

    blocks = bytearray()
    for start in range(0, len(packed), 255):
        block = packed[start : start + 255]
        blocks.append(len(block))
        blocks.extend(block)
    blocks.append(0)
    return bytes(blocks)


def create_demo_gif(reference: np.ndarray, output: Path) -> None:
    width, height = 480, 270
    header = bytearray(b"GIF89a")
    header.extend(struct.pack("<HHBBB", width, height, 0xF7, 0, 0))
    header.extend(_gif_palette())
    header.extend(b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00")
    for step in range(3):
        frame = cv2.resize(_demo_frame(reference, step), (width, height), interpolation=cv2.INTER_AREA)
        header.extend(b"\x21\xf9\x04\x00")
        header.extend(struct.pack("<H", 80))
        header.extend(b"\x00\x00")
        header.extend(b"\x2c")
        header.extend(struct.pack("<HHHHB", 0, 0, width, height, 0))
        header.append(8)
        header.extend(_gif_image_data(frame))
    header.append(0x3B)
    output.write_bytes(header)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing catalog images, including non-synthetic local files",
    )
    args = parser.parse_args()

    inventory = pd.read_csv(INVENTORY, dtype={"sku": str}).fillna("")
    manifest_path = IMAGE_DIR / "manifest.json"
    previous = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("entries", {})
    existing_images = list(IMAGE_DIR.glob("*.jpg"))
    non_synthetic = [
        path.name
        for path in existing_images
        if previous.get(path.stem, {}).get("source") != "synthetic"
    ]
    if non_synthetic and not args.force:
        raise SystemExit(
            "Refusing to overwrite existing non-synthetic catalog images. "
            "Use --force only if replacing them is intended."
        )

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    entries = {}
    grouped_rows = inventory.groupby("image_url", sort=True, dropna=False)
    for _, group in grouped_rows:
        rows = group.to_dict(orient="records")
        first = rows[0]
        members = sorted(str(row["sku"]) for row in rows)
        image = draw_jewelry(
            first["item_type"],
            first["metal"],
            first["gemstone"],
            seed=sum(map(ord, str(first["sku"]))),
        )
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError(f"Could not encode synthetic image for {first['sku']}")
        raw = encoded.tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        for sku in members:
            path = IMAGE_DIR / f"{sku}.jpg"
            path.write_bytes(raw)
            entries[sku] = {
                "source": "synthetic",
                "status": "synthetic",
                "bytes": len(raw),
                "sha256": digest,
                "image_is_unique": len(members) == 1,
                "shared_with": [member for member in members if member != sku],
                "notes": "Generated illustration; not a photograph of a catalog item.",
            }

    manifest_path.write_text(
        json.dumps({"entries": entries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reference = draw_jewelry("ring", "gold", "diamond", seed=18)
    reference_path = MEDIA_DIR / "reference-ring.png"
    if not cv2.imwrite(str(reference_path), reference):
        raise RuntimeError(f"Could not write {reference_path}")
    video_path = MEDIA_DIR / "demo.mp4"
    create_demo_video(reference, video_path)
    gif_path = MEDIA_DIR / "demo.gif"
    create_demo_gif(reference, gif_path)
    print(f"Generated {len(entries)} synthetic catalog illustrations in {IMAGE_DIR}")
    print(f"Generated synthetic reference image: {reference_path}")
    print(f"Generated walkthrough GIF and video: {gif_path}, {video_path}")


if __name__ == "__main__":
    main()
