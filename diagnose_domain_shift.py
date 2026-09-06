from pathlib import Path

import numpy as np

from verishield_ai.forensic.features import FEATURE_NAMES, extract_features


SYNTHETIC = Path("dataset/generated/authentic")
FUNSD = Path("dataset/external/FUNSD/dataset/testing_data/images")


def extract_directory(directory, extensions):
    files = sorted(
        p for p in directory.glob("*")
        if p.suffix.lower() in extensions
    )

    vectors = []

    for path in files:
        vector, _ = extract_features(str(path))
        vectors.append(vector)

    return np.asarray(vectors, dtype=np.float32)


synthetic = extract_directory(
    SYNTHETIC,
    {".jpg", ".jpeg", ".png"}
)

funsd = extract_directory(
    FUNSD,
    {".png"}
)

synthetic_mean = synthetic.mean(axis=0)
funsd_mean = funsd.mean(axis=0)

synthetic_std = synthetic.std(axis=0) + 1e-6

standardized_shift = np.abs(
    funsd_mean - synthetic_mean
) / synthetic_std

print("=== SYNTHETIC vs FUNSD DOMAIN SHIFT ===")
print(f"Synthetic authentic samples: {len(synthetic)}")
print(f"FUNSD authentic samples: {len(funsd)}")
print(f"Feature count: {len(FEATURE_NAMES)}")

print("\nFeature comparison:")

for name, sm, fm, shift in zip(
    FEATURE_NAMES,
    synthetic_mean,
    funsd_mean,
    standardized_shift,
):
    print(
        f"{name}: "
        f"synthetic_mean={sm:.4f}, "
        f"funsd_mean={fm:.4f}, "
        f"standardized_shift={shift:.4f}"
    )

print("\nOverall mean standardized shift:")
print(f"{standardized_shift.mean():.4f}")

print("Maximum standardized shift:")
print(f"{standardized_shift.max():.4f}")

print("Features with shift > 2:")
print(int(np.sum(standardized_shift > 2)))
