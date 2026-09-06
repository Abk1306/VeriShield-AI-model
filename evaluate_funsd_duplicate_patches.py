from pathlib import Path

import cv2
import numpy as np


SOURCE = Path("dataset/external/FUNSD/dataset/testing_data/images")


def detect_duplicate_patch(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    h, w = image.shape

    patch_h = max(40, int(h * 0.08))
    patch_w = max(40, int(w * 0.20))

    step_y = max(20, patch_h // 2)
    step_x = max(20, patch_w // 2)

    patches = []

    for y in range(0, h - patch_h + 1, step_y):
        for x in range(0, w - patch_w + 1, step_x):
            patch = image[y:y + patch_h, x:x + patch_w]

            patch = cv2.resize(patch, (64, 32))
            patch = patch.astype(np.float32)
            patch = (patch - patch.mean()) / (patch.std() + 1e-6)

            patches.append((x, y, patch))

    best_similarity = -1.0

    for i in range(len(patches)):
        x1, y1, p1 = patches[i]

        for j in range(i + 1, len(patches)):
            x2, y2, p2 = patches[j]

            if (
                abs(x1 - x2) < patch_w * 0.75
                and abs(y1 - y2) < patch_h * 0.75
            ):
                continue

            similarity = float(np.mean(p1 * p2))

            if similarity > best_similarity:
                best_similarity = similarity

    return best_similarity


files = sorted(SOURCE.glob("*.png"))

scores = []

print("=== FUNSD AUTHENTIC DUPLICATE-PATCH TEST ===")

for path in files:
    score = detect_duplicate_patch(path)
    scores.append(score)

    print(f"{path.name}: similarity={score:.4f}")

print("\n=== SUMMARY ===")
print(f"Documents: {len(scores)}")
print(f"Minimum: {min(scores):.4f}")
print(f"Maximum: {max(scores):.4f}")
print(f"Mean: {np.mean(scores):.4f}")
print(f"Median: {np.median(scores):.4f}")
print(f"Above 0.65: {sum(s >= 0.65 for s in scores)}")
print(f"Above 0.75: {sum(s >= 0.75 for s in scores)}")
print(f"Above 0.85: {sum(s >= 0.85 for s in scores)}")
print(f"Above 0.90: {sum(s >= 0.90 for s in scores)}")
