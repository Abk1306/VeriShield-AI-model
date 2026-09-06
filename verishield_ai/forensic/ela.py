from pathlib import Path

import cv2
import numpy as np


def compute_ela(
    image: np.ndarray,
    quality: int = 90,
) -> np.ndarray:
    """
    Compute an Error Level Analysis (ELA) map.

    The image is JPEG-compressed at the specified quality and the
    absolute pixel difference from the original is calculated.

    Args:
        image: OpenCV BGR image.
        quality: JPEG compression quality from 1 to 100.

    Returns:
        Grayscale ELA map normalized to 0-255.

    Raises:
        ValueError: If the image is invalid or quality is outside 1-100.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image.")

    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100.")

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )

    if not success:
        raise ValueError("Unable to JPEG-compress image for ELA.")

    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if recompressed is None:
        raise ValueError("Unable to decode recompressed image.")

    difference = cv2.absdiff(image, recompressed)

    ela = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    max_value = float(ela.max())

    if max_value > 0:
        ela = cv2.convertScaleAbs(
            ela,
            alpha=255.0 / max_value,
        )
    else:
        ela = np.zeros_like(ela)

    return ela


def ela_statistics(ela_map: np.ndarray) -> dict:
    """
    Calculate summary statistics for an ELA map.
    """
    if ela_map is None or ela_map.size == 0:
        raise ValueError("Invalid or empty ELA map.")

    return {
        "mean": float(np.mean(ela_map)),
        "std": float(np.std(ela_map)),
        "max": int(np.max(ela_map)),
        "high_error_ratio": float(
            np.mean(ela_map >= 200)
        ),
    }


def save_ela_map(
    ela_map: np.ndarray,
    output_path: str,
) -> str:
    """
    Save an ELA map to disk.
    """
    if ela_map is None or ela_map.size == 0:
        raise ValueError("Invalid or empty ELA map.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output), ela_map):
        raise ValueError(f"Unable to save ELA map: {output_path}")

    return str(output)
