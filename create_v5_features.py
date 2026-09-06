from pathlib import Path

import cv2
import numpy as np

from verishield_ai.forensic.ela import compute_ela
from verishield_ai.forensic.noise import compute_noise_map
from verishield_ai.forensic.compression import compute_compression_map
from verishield_ai.forensic.features import extract_features


V5_FEATURE_NAMES = []


def _grid_statistics(map_image, grid_size=4):
    """
    Divide a forensic map into a grid and calculate
    local statistics for every cell.
    """
    h, w = map_image.shape[:2]
    features = []

    for row in range(grid_size):
        y1 = row * h // grid_size
        y2 = (row + 1) * h // grid_size

        for col in range(grid_size):
            x1 = col * w // grid_size
            x2 = (col + 1) * w // grid_size

            region = map_image[y1:y2, x1:x2]

            if region.size == 0:
                features.extend([0.0, 0.0, 0.0])
                continue

            features.extend([
                float(np.mean(region)),
                float(np.std(region)),
                float(np.percentile(region, 95)),
            ])

    return features


def _local_inconsistency(map_image, grid_size=4):
    """
    Measure how different each local region is from
    the overall document distribution.
    """
    h, w = map_image.shape[:2]
    global_mean = float(np.mean(map_image))
    global_std = float(np.std(map_image)) + 1e-6

    features = []

    for row in range(grid_size):
        y1 = row * h // grid_size
        y2 = (row + 1) * h // grid_size

        for col in range(grid_size):
            x1 = col * w // grid_size
            x2 = (col + 1) * w // grid_size

            region = map_image[y1:y2, x1:x2]

            if region.size == 0:
                features.extend([0.0, 0.0])
                continue

            local_mean = float(np.mean(region))
            local_std = float(np.std(region))

            mean_difference = abs(local_mean - global_mean) / global_std
            std_difference = abs(local_std - global_std) / global_std

            features.extend([
                mean_difference,
                std_difference,
            ])

    return features


def extract_v5_features(file_path):
    """
    Extract the original 16 forensic features plus
    spatial/local forensic features.

    Returns:
        feature_vector, feature_names
    """
    original_features, original_names = extract_features(file_path)

    image = cv2.imread(str(file_path))

    if image is None:
        raise ValueError(f"Could not read image: {file_path}")

    ela_map = compute_ela(image)
    noise_map = compute_noise_map(image)
    compression_map = compute_compression_map(image)

    spatial_features = []
    spatial_names = []

    maps = {
        "ela": ela_map,
        "noise": noise_map,
        "compression": compression_map,
    }

    for map_name, forensic_map in maps.items():

        grid_features = _grid_statistics(
            forensic_map,
            grid_size=4,
        )

        for index in range(16):
            spatial_names.extend([
                f"{map_name}_region_{index}_mean",
                f"{map_name}_region_{index}_std",
                f"{map_name}_region_{index}_p95",
            ])

        spatial_features.extend(grid_features)

        inconsistency_features = _local_inconsistency(
            forensic_map,
            grid_size=4,
        )

        for index in range(16):
            spatial_names.extend([
                f"{map_name}_region_{index}_mean_inconsistency",
                f"{map_name}_region_{index}_std_inconsistency",
            ])

        spatial_features.extend(inconsistency_features)

    vector = np.concatenate(
        [
            original_features.astype(np.float32),
            np.asarray(spatial_features, dtype=np.float32),
        ]
    )

    names = list(original_names) + spatial_names

    if len(names) != len(vector):
        raise RuntimeError(
            f"Feature/name mismatch: {len(vector)} vs {len(names)}"
        )

    if not np.all(np.isfinite(vector)):
        raise RuntimeError(
            f"Non-finite V5 features detected for {file_path}"
        )

    return vector, names


if __name__ == "__main__":
    sample = Path("dataset/authentic/test_document.png")

    features, names = extract_v5_features(str(sample))

    print("=== VERISHIELD V5 FEATURE TEST ===")
    print(f"Document: {sample}")
    print(f"Feature count: {len(features)}")
    print(f"Feature names: {len(names)}")
    print(f"Finite features: {np.isfinite(features).all()}")
    print("Status: PASS")
