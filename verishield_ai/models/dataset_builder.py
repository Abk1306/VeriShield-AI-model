from pathlib import Path

import numpy as np

from verishield_ai.forensic.features import extract_features


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def collect_images(directory: str) -> list[Path]:
    """
    Collect supported image files recursively from a directory.
    """
    root = Path(directory)

    if not root.exists():
        return []

    if not root.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_feature_dataset(
    authentic_dir: str,
    tampered_dir: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build an ML dataset from authentic and tampered documents.

    Labels:
        0 = authentic
        1 = tampered
    """
    authentic_files = collect_images(authentic_dir)
    tampered_files = collect_images(tampered_dir)

    if not authentic_files:
        raise ValueError(
            f"No authentic images found in: {authentic_dir}"
        )

    if not tampered_files:
        raise ValueError(
            f"No tampered images found in: {tampered_dir}"
        )

    features = []
    labels = []
    paths = []

    for file_path in authentic_files:
        vector, _ = extract_features(str(file_path))
        features.append(vector)
        labels.append(0)
        paths.append(str(file_path))

    for file_path in tampered_files:
        vector, _ = extract_features(str(file_path))
        features.append(vector)
        labels.append(1)
        paths.append(str(file_path))

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    return X, y, paths
