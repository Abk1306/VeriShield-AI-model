import numpy as np

from verishield_ai.preprocessing import preprocess_document
from verishield_ai.forensic.ela import compute_ela, ela_statistics
from verishield_ai.forensic.noise import compute_noise_map, noise_statistics
from verishield_ai.forensic.compression import (
    compute_compression_map,
    compression_statistics,
)
from verishield_ai.forensic.metadata import (
    analyze_metadata,
    metadata_statistics,
)


FEATURE_NAMES = [
    "ela_mean",
    "ela_std",
    "ela_max",
    "ela_high_error_ratio",
    "noise_mean",
    "noise_std",
    "noise_max",
    "noise_high_noise_ratio",
    "compression_mean",
    "compression_std",
    "compression_max",
    "compression_high_artifact_ratio",
    "has_exif",
    "has_software_tag",
    "has_camera_make",
    "has_camera_model",
]


def extract_features(file_path: str) -> tuple[np.ndarray, list[str]]:
    """
    Extract a fixed numerical feature vector from a document.
    """
    image = preprocess_document(file_path)

    ela_map = compute_ela(image)
    noise_map = compute_noise_map(image)
    compression_map = compute_compression_map(image)

    ela = ela_statistics(ela_map)
    noise = noise_statistics(noise_map)
    compression = compression_statistics(compression_map)

    metadata = metadata_statistics(
        analyze_metadata(file_path)
    )

    features = [
        ela["mean"],
        ela["std"],
        ela["max"],
        ela["high_error_ratio"],
        noise["mean"],
        noise["std"],
        noise["max"],
        noise["high_noise_ratio"],
        compression["mean"],
        compression["std"],
        compression["max"],
        compression["high_artifact_ratio"],
        int(metadata["has_exif"]),
        int(metadata["has_software_tag"]),
        int(metadata["has_camera_make"]),
        int(metadata["has_camera_model"]),
    ]

    return np.asarray(features, dtype=np.float32), FEATURE_NAMES.copy()
