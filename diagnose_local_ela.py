from pathlib import Path

import cv2
import numpy as np

from verishield_ai.forensic.ela import compute_ela


AUTHENTIC = Path("dataset/external/FUNSD/dataset/testing_data/images")
TAMPERED = Path("dataset/external/FUNSD_eval/tampered")


def region_stats(ela, x, y, w, h):
    region = ela[y:y + h, x:x + w]

    return {
        "mean": float(np.mean(region)),
        "std": float(np.std(region)),
        "p95": float(np.percentile(region, 95)),
        "max": float(np.max(region)),
    }


print("=== LOCAL ELA DIAGNOSTIC ===")

for tampered_path in sorted(TAMPERED.glob("*.png")):
    original_name = tampered_path.name.replace("_copy_move", "")
    original_path = AUTHENTIC / original_name

    if not original_path.exists():
        continue

    original_image = cv2.imread(str(original_path))
    tampered_image = cv2.imread(str(tampered_path))

    h, w = tampered_image.shape[:2]

    # Same destination used by create_funsd_eval.py
    source_w = int(w * 0.20)
    source_h = int(h * 0.08)

    dst_x = int(w * 0.60)
    dst_y = int(h * 0.65)

    original_ela = compute_ela(original_image)
    tampered_ela = compute_ela(tampered_image)

    original_stats = region_stats(
        original_ela,
        dst_x,
        dst_y,
        source_w,
        source_h,
    )

    tampered_stats = region_stats(
        tampered_ela,
        dst_x,
        dst_y,
        source_w,
        source_h,
    )

    print(f"\n{tampered_path.name}")
    print(
        f"Original destination: "
        f"mean={original_stats['mean']:.2f}, "
        f"p95={original_stats['p95']:.2f}, "
        f"max={original_stats['max']:.2f}"
    )
    print(
        f"Tampered destination: "
        f"mean={tampered_stats['mean']:.2f}, "
        f"p95={tampered_stats['p95']:.2f}, "
        f"max={tampered_stats['max']:.2f}"
    )
