"""Generate repeatable contour-count fixtures used by the evaluation harness.

The first group draws well-separated circles, which the baseline contours handle
trivially. The second group deliberately breaks that baseline so the reported count
error reflects a real detector rather than an easy fixture:

``touching``     overlapping circles merge into one external contour (undercount)
``gradient``     an uneven background defeats a single global threshold (overcount)
``specks``       small debris survives the area filter and inflates the count
``occluded``     a shape cut in half by the frame border (undercount)
"""

from pathlib import Path

import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "images"


def make_image(component_count: int) -> None:
    """Draw separated black circles on white, one image for each known count."""
    image = np.full((240, 240, 3), 255, dtype=np.uint8)
    positions = [(45, 45), (120, 45), (195, 45), (82, 135), (158, 135)]
    for center in positions[:component_count]:
        cv2.circle(image, center, 20, (0, 0, 0), thickness=-1)
    cv2.imwrite(str(OUTPUT_DIR / f"{component_count}_components.png"), image)


def make_touching() -> None:
    """Five circles in a row, each overlapping its neighbour."""
    image = np.full((240, 240, 3), 255, dtype=np.uint8)
    for center in [(45, 120), (80, 120), (115, 120), (150, 120), (185, 120)]:
        cv2.circle(image, center, 25, (0, 0, 0), thickness=-1)
    cv2.imwrite(str(OUTPUT_DIR / "hard_touching.png"), image)


def make_gradient() -> None:
    """Four circles on a background that darkens from left to right."""
    column = np.linspace(255, 90, 240, dtype=np.uint8)
    background = np.repeat(column[np.newaxis, :], 240, axis=0)
    image = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    for center in [(55, 70), (130, 70), (60, 170), (150, 165)]:
        cv2.circle(image, center, 22, (30, 30, 30), thickness=-1)
    cv2.imwrite(str(OUTPUT_DIR / "hard_gradient.png"), image)


def make_specks() -> None:
    """Three circles plus scattered debris just above the minimum area filter."""
    image = np.full((240, 240, 3), 255, dtype=np.uint8)
    for center in [(70, 70), (160, 70), (115, 175)]:
        cv2.circle(image, center, 24, (0, 0, 0), thickness=-1)
    rng = np.random.default_rng(20260912)
    for _ in range(14):
        x, y = int(rng.integers(15, 225)), int(rng.integers(15, 225))
        cv2.circle(image, (x, y), 3, (40, 40, 40), thickness=-1)
    cv2.imwrite(str(OUTPUT_DIR / "hard_specks.png"), image)


def make_occluded() -> None:
    """Three circles, one of them clipped by the right edge of the frame."""
    image = np.full((240, 240, 3), 255, dtype=np.uint8)
    cv2.circle(image, (50, 80), 26, (0, 0, 0), thickness=-1)
    cv2.circle(image, (130, 80), 26, (0, 0, 0), thickness=-1)
    cv2.circle(image, (235, 175), 30, (0, 0, 0), thickness=-1)
    cv2.imwrite(str(OUTPUT_DIR / "hard_occluded.png"), image)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    for count in range(1, 6):
        make_image(count)
    make_touching()
    make_gradient()
    make_specks()
    make_occluded()