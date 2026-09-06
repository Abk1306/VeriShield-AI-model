from pathlib import Path

import cv2
import numpy as np


AUTHENTIC = Path("dataset/external/FUNSD/dataset/testing_data/images")
TAMPERED = Path("dataset/external/FUNSD_eval/tampered")


def local_noise_inconsistency(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    image = image.astype(np.float32)

    # High-frequency residual.
    blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
    residual = np.abs(image - blurred)

    # Local noise level.
    local_mean = cv2.GaussianBlur(residual, (0, 0), 12.0)

    # Compare each location with its broad neighborhood.
    neighborhood = cv2.GaussianBlur(local_mean, (0, 0), 35.0)

    inconsistency = np.abs(local_mean - neighborhood)

    # Ignore extreme document borders.
    h, w = inconsistency.shape
    margin_y = int(h * 0.03)
    margin_x = int(w * 0.03)

    cropped = inconsistency[
        margin_y:h - margin_y,
        margin_x:w - margin_x,
    ]

    return {
        "mean": float(np.mean(cropped)),
        "std": float(np.std(cropped)),
        "max": float(np.max(cropped)),
        "p95": float(np.percentile(cropped, 95)),
        "p99": float(np.percentile(cropped, 99)),
    }


def evaluate(directory):
    values = []

    for path in sorted(directory.glob("*.png")):
        stats = local_noise_inconsistency(path)
        values.append(stats)

    return values


authentic = evaluate(AUTHENTIC)
tampered = evaluate(TAMPERED)

print("=== LOCAL NOISE INCONSISTENCY ===")

print("\nAUTHENTIC FUNSD")
print(f"Documents: {len(authentic)}")
print(f"Mean: {np.mean([x['mean'] for x in authentic]):.4f}")
print(f"95th percentile mean: {np.mean([x['p95'] for x in authentic]):.4f}")
print(f"99th percentile mean: {np.mean([x['p99'] for x in authentic]):.4f}")
print(f"Maximum p99: {max(x['p99'] for x in authentic):.4f}")

print("\nTAMPERED FUNSD")
print(f"Documents: {len(tampered)}")
print(f"Mean: {np.mean([x['mean'] for x in tampered]):.4f}")
print(f"95th percentile mean: {np.mean([x['p95'] for x in tampered]):.4f}")
print(f"99th percentile mean: {np.mean([x['p99'] for x in tampered]):.4f}")
print(f"Maximum p99: {max(x['p99'] for x in tampered):.4f}")
