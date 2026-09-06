from pathlib import Path

import cv2
import numpy as np


def create_anomaly_map(
    ela_map: np.ndarray,
    noise_map: np.ndarray,
    compression_map: np.ndarray,
) -> np.ndarray:
    """
    Combine forensic maps into a single anomaly map.

    Weights:
        ELA: 45%
        Noise: 35%
        Compression: 20%
    """
    if ela_map.shape != noise_map.shape or ela_map.shape != compression_map.shape:
        raise ValueError("All forensic maps must have the same shape.")

    ela = ela_map.astype(np.float32)
    noise = noise_map.astype(np.float32)
    compression = compression_map.astype(np.float32)

    combined = (
        0.45 * ela
        + 0.35 * noise
        + 0.20 * compression
    )

    return np.clip(combined, 0, 255).astype(np.uint8)


def detect_suspicious_regions(
    anomaly_map: np.ndarray,
    threshold: int = 80,
    min_area: int = 200,
    max_area_ratio: float = 0.20,
    max_regions: int = 10,
) -> list[dict]:
    """
    Detect and rank suspicious regions from an anomaly map.

    Parameters:
        anomaly_map: 8-bit anomaly map.
        threshold: anomaly threshold used to create candidate regions.
        min_area: minimum connected-component area.
        max_area_ratio: ignore regions larger than this fraction of image area.
        max_regions: maximum number of strongest regions returned.
    """
    if anomaly_map.ndim != 2:
        raise ValueError("anomaly_map must be a 2D grayscale array.")

    height, width = anomaly_map.shape
    image_area = height * width

    binary = np.where(
        anomaly_map >= threshold,
        255,
        0,
    ).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    binary = cv2.dilate(
        binary,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    regions = []

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < min_area:
            continue

        if area > image_area * max_area_ratio:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        roi = anomaly_map[y:y + h, x:x + w]

        if roi.size == 0:
            continue

        mean_anomaly = float(np.mean(roi)) / 255.0
        peak_anomaly = float(np.max(roi)) / 255.0

        area_ratio = area / image_area
        size_score = min(area_ratio * 100.0, 1.0)

        confidence = (
            0.45 * mean_anomaly
            + 0.35 * peak_anomaly
            + 0.20 * size_score
        )

        regions.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area": round(area, 2),
                "confidence": round(float(confidence), 4),
            }
        )

    regions.sort(
        key=lambda region: region["confidence"],
        reverse=True,
    )

    return regions[:max_regions]


def save_anomaly_map(
    anomaly_map: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Save an anomaly map as a grayscale image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), anomaly_map):
        raise IOError(
            f"Failed to save anomaly map: {output_path}"
        )
