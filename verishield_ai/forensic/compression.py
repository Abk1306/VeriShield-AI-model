import cv2
import numpy as np


def compute_compression_map(image: np.ndarray) -> np.ndarray:
    """
    Estimate block-level compression artifacts in a document image.

    JPEG compression operates on 8x8 blocks. This function measures
    intensity discontinuities around those block boundaries and
    produces a normalized grayscale artifact map.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    height, width = gray.shape

    vertical_diff = np.zeros_like(gray)
    horizontal_diff = np.zeros_like(gray)

    if width > 1:
        vertical_diff[:, 1:] = np.abs(
            gray[:, 1:] - gray[:, :-1]
        )

    if height > 1:
        horizontal_diff[1:, :] = np.abs(
            gray[1:, :] - gray[:-1, :]
        )

    # Examine differences specifically around 8x8 JPEG block boundaries.
    boundary_map = np.zeros_like(gray)

    for x in range(8, width, 8):
        boundary_map[:, x] = vertical_diff[:, x]

    for y in range(8, height, 8):
        boundary_map[y, :] = np.maximum(
            boundary_map[y, :],
            horizontal_diff[y, :],
        )

    # Spread boundary evidence into neighboring pixels.
    boundary_map = cv2.GaussianBlur(
        boundary_map,
        (5, 5),
        0,
    )

    max_value = float(boundary_map.max())

    if max_value > 0:
        boundary_map = cv2.convertScaleAbs(
            boundary_map,
            alpha=255.0 / max_value,
        )
    else:
        boundary_map = np.zeros_like(
            boundary_map,
            dtype=np.uint8,
        )

    return boundary_map


def compression_statistics(
    compression_map: np.ndarray,
) -> dict:
    """
    Calculate summary statistics for compression artifacts.
    """
    if compression_map is None or compression_map.size == 0:
        raise ValueError("Invalid or empty compression map.")

    return {
        "mean": float(np.mean(compression_map)),
        "std": float(np.std(compression_map)),
        "max": int(np.max(compression_map)),
        "high_artifact_ratio": float(
            np.mean(compression_map >= 200)
        ),
    }


def save_compression_map(
    compression_map: np.ndarray,
    output_path: str,
) -> str:
    """
    Save a compression artifact map to disk.
    """
    if compression_map is None or compression_map.size == 0:
        raise ValueError("Invalid or empty compression map.")

    if not cv2.imwrite(output_path, compression_map):
        raise ValueError(
            f"Unable to save compression map: {output_path}"
        )

    return output_path
