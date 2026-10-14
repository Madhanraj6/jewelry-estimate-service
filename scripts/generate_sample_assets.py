"""Generate reference-photo walkthrough media.

Run this in the project root with:
    python scripts/generate_sample_assets.py

The script uses the bundled owner-provided reference images under examples/media.
Catalog images are fetched separately from the product URLs in data/invent.csv.
"""

from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MEDIA_DIR = ROOT / "examples" / "media"
INVENTORY_MATCHES = {
    "ring": "RNG-001",
    "chain": "CHN-001",
    "pendant": "PEN-001",
}


def _rounded_card(frame: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int], fill=(255, 255, 255)) -> None:
    cv2.rectangle(frame, top_left, bottom_right, fill, -1, cv2.LINE_AA)
    cv2.rectangle(frame, top_left, bottom_right, (225, 229, 235), 2, cv2.LINE_AA)


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def _demo_frame(
    reference: np.ndarray, candidate: np.ndarray, step: int, item_type: str, sku: str
) -> np.ndarray:
    width, height = 1280, 720
    frame = np.full((height, width, 3), (247, 248, 251), dtype=np.uint8)
    cv2.putText(frame, "JEWELRY ESTIMATE SERVICE", (54, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (70, 82, 98), 2, cv2.LINE_AA)
    captions = [
        ("Reference photo 1", "Ring"),
        ("Reference photo 2", "Chain"),
        ("Reference photo 3", "Pendant"),
    ]
    _, subtitle = captions[step]
    cv2.putText(frame, "Reference photo -> closest catalog match", (54, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (34, 50, 70), 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (54, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 110, 125), 1, cv2.LINE_AA)
    _rounded_card(frame, (54, 235), (584, 646))
    _rounded_card(frame, (626, 235), (1226, 646))
    frame[270:590, 84:554] = _fit_image(reference, 470, 320)
    frame[270:590, 704:1144] = _fit_image(candidate, 440, 320)
    cv2.putText(frame, f"UPLOADED REFERENCE: {item_type.upper()}", (78, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 110, 125), 1, cv2.LINE_AA)
    cv2.putText(frame, f"TOP CATALOG CANDIDATE: {sku}", (652, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 110, 125), 1, cv2.LINE_AA)
    cv2.putText(frame, "Product image from the CSV-linked catalog listing.", (652, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (75, 85, 98), 1, cv2.LINE_AA)
    for index in range(3):
        cv2.circle(frame, (54 + index * 40, 684), 8, (183, 121, 31) if index == step else (215, 220, 228), -1, cv2.LINE_AA)
    return frame


def create_demo_video(
    references: list[tuple[str, np.ndarray, np.ndarray, str]], output: Path
) -> None:
    width, height, fps = 1280, 720, 15
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize an MP4 encoder")
    try:
        for step, (item_type, reference, candidate, sku) in enumerate(references):
            frame = _demo_frame(reference, candidate, step, item_type, sku)
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


def create_demo_gif(
    references: list[tuple[str, np.ndarray, np.ndarray, str]], output: Path
) -> None:
    width, height = 480, 270
    header = bytearray(b"GIF89a")
    header.extend(struct.pack("<HHBBB", width, height, 0xF7, 0, 0))
    header.extend(_gif_palette())
    header.extend(b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00")
    for step, (item_type, reference, candidate, sku) in enumerate(references):
        frame = cv2.resize(
            _demo_frame(reference, candidate, step, item_type, sku),
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
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
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    reference_specs = [
        ("ring", MEDIA_DIR / "reference-ring.jpg"),
        ("chain", MEDIA_DIR / "reference-chain.jpg"),
        ("pendant", MEDIA_DIR / "reference-pendant.jpg"),
    ]
    references = []
    for item_type, path in reference_specs:
        reference = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if reference is None:
            raise FileNotFoundError(f"Required demo reference image is missing or unreadable: {path}")
        sku = INVENTORY_MATCHES[item_type]
        candidate_path = ROOT / "data" / "images" / f"{sku}.jpg"
        candidate = cv2.imread(str(candidate_path), cv2.IMREAD_COLOR)
        if candidate is None:
            raise FileNotFoundError(
                f"Required catalog candidate image is missing or unreadable: {candidate_path}"
            )
        references.append((item_type, reference, candidate, sku))

    video_path = MEDIA_DIR / "demo.mp4"
    create_demo_video(references, video_path)
    gif_path = MEDIA_DIR / "demo.gif"
    create_demo_gif(references, gif_path)
    print(f"Using {len(references)} provided reference photos from {MEDIA_DIR}")
    print(f"Generated walkthrough GIF and video: {gif_path}, {video_path}")


if __name__ == "__main__":
    main()
