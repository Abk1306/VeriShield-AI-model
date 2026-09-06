from pathlib import Path

import cv2
import numpy as np


SOURCE = Path("dataset/external/FUNSD_eval/tampered")


def detect_duplicate_patch(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    h, w = image.shape

    # Compare medium-sized patches on a coarse grid.
    patch_h = max(40, int(h * 0.08))
    patch_w = max(40, int(w * 0.20))

    step_y = max(20, patch_h // 2)
    step_x = max(20, patch_w // 2)

    patches = []

    for y in range(0, h - patch_h + 1, step_y):
        for x in range(0, w - patch_w + 1, step_x):
            patch = image[y:y + patch_h, x:x + patch_w]

            # Normalize to reduce sensitivity to small brightness changes.
            patch = cv2.resize(patch, (64, 32))
            patch = patch.astype(np.float32)
            patch = (patch - patch.mean()) / (patch.std() + 1e-6)

            patches.append((x, y, patch))

    best_similarity = -1.0
    best_pair = None

    for i in range(len(patches)):
        x1, y1, p1 = patches[i]

        for j in range(i + 1, len(patches)):
            x2, y2, p2 = patches[j]

            # Ignore nearby/overlapping patches.
            if abs(x1 - x2) < patch_w * 0.75 and abs(y1 - y2) < patch_h * 0.75:
                continue

            similarity = float(np.mean(p1 * p2))

            if similarity > best_similarity:
                best_similarity = similarity
                best_pair = ((x1, y1), (x2, y2))

    return best_similarity, best_pair


print("=== DIRECT DUPLICATE-PATCH DIAGNOSTIC ===")

for path in sorted(SOURCE.glob("*.png")):
    similarity, pair = detect_duplicate_patch(path)

    print(
        f"{path.name}: "
        f"best_similarity={similarity:.4f}, "
        f"locations={pair}"
    )
