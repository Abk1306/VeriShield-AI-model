import cv2
import numpy as np


def compute_noise_map(image: np.ndarray) -> np.ndarray:
    """
    Estimate local noise/residual variation in a document image.

    Returns a normalized grayscale map from 0-255.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Estimate high-frequency residual using Gaussian smoothing.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = cv2.absdiff(gray, blurred)

    # Smooth the residual to obtain local noise intensity.
    noise = cv2.GaussianBlur(residual, (7, 7), 0)

    max_value = float(noise.max())

    if max_value > 0:
        noise = cv2.convertScaleAbs(
            noise,
            alpha=255.0 / max_value,
        )
    else:
        noise = np.zeros_like(noise)

    return noise


def noise_statistics(noise_map: np.ndarray) -> dict:
    """
    Calculate summary statistics for a noise map.
    """
    if noise_map is None or noise_map.size == 0:
        raise ValueError("Invalid or empty noise map.")

    return {
        "mean": float(np.mean(noise_map)),
        "std": float(np.std(noise_map)),
        "max": int(np.max(noise_map)),
        "high_noise_ratio": float(
            np.mean(noise_map >= 200)
        ),
    }


def save_noise_map(
    noise_map: np.ndarray,
    output_path: str,
) -> str:
    """
    Save a noise analysis map to disk.
    """
    if noise_map is None or noise_map.size == 0:
        raise ValueError("Invalid or empty noise map.")

    if not cv2.imwrite(output_path, noise_map):
        raise ValueError(
            f"Unable to save noise map: {output_path}"
        )

    return output_path
