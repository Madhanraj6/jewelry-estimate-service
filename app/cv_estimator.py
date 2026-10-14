"""Simple, explainable OpenCV component counter for jewelry photographs."""

import base64
from pathlib import Path

import cv2
import numpy as np


def estimate_components(image_input: str | Path, minimum_area: float = 25.0) -> dict:
    """Count distinct foreground contours in an image.

    This is a deliberately simple simulation of gemstone/component counting; it is
    not a production-grade gemstone detector and should feed review, not pricing.
    """
    image = _load_image(image_input)
    if image is None:
        raise ValueError("Could not read the supplied image")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, threshold = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    component_count = sum(cv2.contourArea(contour) >= minimum_area for contour in contours)

    return {
        "component_count": int(component_count),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
    }


def _load_image(image_input: str | Path) -> np.ndarray | None:
    """Load either a local path (evaluation) or browser data URL (frontend)."""
    value = str(image_input)
    if value.startswith("data:image/"):
        try:
            _, encoded = value.split(",", 1)
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) > 5 * 1024 * 1024:
                raise ValueError("Image must be 5 MB or smaller")
            return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError("Image upload is not valid base64 data") from exc

    return cv2.imread(str(Path(value)))
