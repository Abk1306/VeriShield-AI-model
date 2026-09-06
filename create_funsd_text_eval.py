from pathlib import Path
import json

import cv2
import numpy as np


SOURCE_IMAGES = Path(
    "dataset/external/FUNSD/dataset/testing_data/images"
)

SOURCE_ANNOTATIONS = Path(
    "dataset/external/FUNSD/dataset/testing_data/annotations"
)

OUTPUT = Path(
    "dataset/external/FUNSD_eval/text_edit"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

files = sorted(SOURCE_IMAGES.glob("*.png"))[:10]

created = 0

for image_path in files:
    annotation_path = SOURCE_ANNOTATIONS / f"{image_path.stem}.json"

    if not annotation_path.exists():
        print(f"SKIPPED annotation: {image_path.name}")
        continue

    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    form = data.get("form", [])

    # Find a reasonably sized text region.
    candidates = []

    for item in form:
        box = item.get("box")

        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        if width >= 25 and height >= 12:
            candidates.append((box, item.get("text", "")))

    if not candidates:
        print(f"SKIPPED no suitable text box: {image_path.name}")
        continue

    # Use the first suitable real text region.
    (x1, y1, x2, y2), original_text = candidates[0]

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"SKIPPED unreadable: {image_path.name}")
        continue

    # Add a small opaque rectangular alteration inside the
    # real annotated text region.
    margin_x = max(2, int((x2 - x1) * 0.10))
    margin_y = max(2, int((y2 - y1) * 0.10))

    tx1 = x1 + margin_x
    ty1 = y1 + margin_y
    tx2 = x2 - margin_x
    ty2 = y2 - margin_y

    # Estimate local background from the region border.
    border = image[
        max(0, y1 - 2):min(image.shape[0], y2 + 2),
        max(0, x1 - 2):min(image.shape[1], x2 + 2),
    ]

    background = np.median(border.reshape(-1, 3), axis=0).astype(np.uint8)

    cv2.rectangle(
        image,
        (tx1, ty1),
        (tx2, ty2),
        tuple(int(v) for v in background),
        thickness=-1,
    )

    # Add replacement text-like marks.
    cv2.putText(
        image,
        "EDIT",
        (tx1, min(ty2 - 2, ty1 + max(10, ty2 - ty1 - 2))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    output_path = OUTPUT / f"{image_path.stem}_text_edit.png"
    cv2.imwrite(str(output_path), image)

    print(
        f"CREATED: {output_path} | "
        f"original_text={original_text!r} | "
        f"box={[x1, y1, x2, y2]}"
    )

    created += 1

print(f"\nCreated {created} FUNSD text-edit evaluation images.")
