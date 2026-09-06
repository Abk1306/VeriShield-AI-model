from pathlib import Path

import cv2
import numpy as np


SOURCE = Path("dataset/external/FUNSD/dataset/testing_data/images")
OUTPUT = Path("dataset/external/FUNSD_eval/tampered")
OUTPUT.mkdir(parents=True, exist_ok=True)

files = sorted(SOURCE.glob("*.png"))[:10]

for index, path in enumerate(files):
    image = cv2.imread(str(path))

    if image is None:
        print(f"SKIPPED: {path.name}")
        continue

    height, width = image.shape[:2]

    # Create a controlled copy-move tampering:
    # copy a document region and paste it into another location.
    source_w = int(width * 0.20)
    source_h = int(height * 0.08)

    src_x = int(width * 0.10)
    src_y = int(height * 0.25)

    dst_x = int(width * 0.60)
    dst_y = int(height * 0.65)

    source_region = image[
        src_y:src_y + source_h,
        src_x:src_x + source_w
    ].copy()

    image[
        dst_y:dst_y + source_h,
        dst_x:dst_x + source_w
    ] = source_region

    output_path = OUTPUT / f"{path.stem}_copy_move.png"
    cv2.imwrite(str(output_path), image)

    print(f"CREATED: {output_path}")

print(f"\nCreated {len(list(OUTPUT.glob('*.png')))} FUNSD tampered evaluation images.")
