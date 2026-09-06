from pathlib import Path

import cv2
import numpy as np

from verishield_ai.forensic.ela import compute_ela
from verishield_ai.forensic.noise import compute_noise_map
from verishield_ai.forensic.compression import compute_compression_map


def spatial_features(path, grid_rows=4, grid_cols=4):
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Could not read: {path}")

    ela = compute_ela(image)
    noise = compute_noise_map(image)
    compression = compute_compression_map(image)

    h, w = ela.shape

    features = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            y1 = row * h // grid_rows
            y2 = (row + 1) * h // grid_rows

            x1 = col * w // grid_cols
            x2 = (col + 1) * w // grid_cols

            for anomaly_map in (ela, noise, compression):
                region = anomaly_map[y1:y2, x1:x2].astype(np.float32)

                features.extend(
                    [
                        float(np.mean(region)),
                        float(np.std(region)),
                        float(np.percentile(region, 95)),
                        float(np.percentile(region, 99)),
                        float(np.max(region)),
                    ]
                )

    return np.asarray(features, dtype=np.float32)


AUTHENTIC = Path("dataset/external/FUNSD/dataset/testing_data/images")
TAMPERED = Path("dataset/external/FUNSD_eval/tampered")

authentic_files = sorted(AUTHENTIC.glob("*.png"))
tampered_files = sorted(TAMPERED.glob("*.png"))

authentic_vectors = np.asarray(
    [spatial_features(p) for p in authentic_files]
)

tampered_vectors = np.asarray(
    [spatial_features(p) for p in tampered_files]
)

print("=== SPATIAL FEATURE DIAGNOSTIC ===")
print(f"Authentic documents: {len(authentic_vectors)}")
print(f"Tampered documents: {len(tampered_vectors)}")
print(f"Features per document: {authentic_vectors.shape[1]}")

# Compare each tampered document against the distribution
# of authentic FUNSD documents.
auth_mean = authentic_vectors.mean(axis=0)
auth_std = authentic_vectors.std(axis=0) + 1e-6

tampered_z = np.abs(
    (tampered_vectors - auth_mean) / auth_std
)

authentic_z = np.abs(
    (authentic_vectors - auth_mean) / auth_std
)

auth_scores = authentic_z.mean(axis=1)
tampered_scores = tampered_z.mean(axis=1)

print("\nAverage normalized spatial anomaly:")
print(f"Authentic mean: {auth_scores.mean():.4f}")
print(f"Authentic max:  {auth_scores.max():.4f}")
print(f"Tampered mean:  {tampered_scores.mean():.4f}")
print(f"Tampered min:   {tampered_scores.min():.4f}")
print(f"Tampered max:   {tampered_scores.max():.4f}")

print("\nPer-document tampered scores:")

for path, score in zip(tampered_files, tampered_scores):
    print(f"{path.name}: {score:.4f}")
