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


def _rounded_card(frame: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int], fill=(255, 255, 255)) -> None:
    cv2.rectangle(frame, top_left, bottom_right, fill, -1, cv2.LINE_AA)
    cv2.rectangle(frame, top_left, bottom_right, (225, 229, 235), 2, cv2.LINE_AA)


def _demo_frame(reference: np.ndarray, step: int, item_type: str) -> np.ndarray:
    width, height = 1280, 720
    frame = np.full((height, width, 3), (247, 248, 251), dtype=np.uint8)
    cv2.putText(frame, "JEWELRY ESTIMATE SERVICE", (54, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (70, 82, 98), 2, cv2.LINE_AA)
    captions = [
        ("Reference photo 1", "Ring"),
        ("Reference photo 2", "Chain"),
        ("Reference photo 3", "Pendant"),
    ]
    title, subtitle = captions[step]
    cv2.putText(frame, title, (54, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (34, 50, 70), 3, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (54, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 110, 125), 1, cv2.LINE_AA)
    _rounded_card(frame, (54, 235), (584, 646))
    _rounded_card(frame, (626, 235), (1226, 646))
    preview = cv2.resize(reference, (450, 450), interpolation=cv2.INTER_AREA)
    frame[270:720, 94:544] = preview
    cv2.putText(frame, f"UPLOADED REFERENCE: {item_type.upper()}", (98, 616), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 110, 125), 1, cv2.LINE_AA)
    details = [
        ([f"Reference category: {item_type}", "Customer image shown as submitted.", "No extraction inferred here."], (45, 80, 145)),
        (["Use with a short product brief.", "Model output is schema-validated.", "Unknowns stay explicitly unknown."], (45, 115, 75)),
        (["Catalog candidates are ranked", "with deterministic scoring.", "Low confidence means review.", "Catalog base price is not a quote."], (45, 80, 145)),
    ]
    lines, color = details[step]
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (678, 340 + index * 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    for index in range(3):
        cv2.circle(frame, (54 + index * 40, 684), 8, (183, 121, 31) if index == step else (215, 220, 228), -1, cv2.LINE_AA)
    return frame


def create_demo_video(references: list[tuple[str, np.ndarray]], output: Path) -> None:
    width, height, fps = 1280, 720, 15
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize an MP4 encoder")
    try:
        for step, (item_type, reference) in enumerate(references):
            frame = _demo_frame(reference, step, item_type)
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


def create_demo_gif(references: list[tuple[str, np.ndarray]], output: Path) -> None:
    width, height = 480, 270
    header = bytearray(b"GIF89a")
    header.extend(struct.pack("<HHBBB", width, height, 0xF7, 0, 0))
    header.extend(_gif_palette())
    header.extend(b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00")
    for step, (item_type, reference) in enumerate(references):
        frame = cv2.resize(
            _demo_frame(reference, step, item_type),
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
        references.append((item_type, reference))

    video_path = MEDIA_DIR / "demo.mp4"
    create_demo_video(references, video_path)
    gif_path = MEDIA_DIR / "demo.gif"
    create_demo_gif(references, gif_path)
    print(f"Using {len(references)} provided reference photos from {MEDIA_DIR}")
    print(f"Generated walkthrough GIF and video: {gif_path}, {video_path}")


if __name__ == "__main__":
    main()
