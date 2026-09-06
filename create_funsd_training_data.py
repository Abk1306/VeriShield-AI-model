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

ROOT = Path("dataset/external/FUNSD_train_eval")

AUTHENTIC = ROOT / "authentic"
TEXT = ROOT / "tampered_text"
COPY_MOVE = ROOT / "tampered_copy_move"
RECOMPRESSION = ROOT / "tampered_recompression"
BLUR = ROOT / "tampered_blur"

for directory in [AUTHENTIC, TEXT, COPY_MOVE, RECOMPRESSION, BLUR]:
    directory.mkdir(parents=True, exist_ok=True)


files = sorted(SOURCE_IMAGES.glob("*.png"))[:40]

created = {
    "authentic": 0,
    "text": 0,
    "copy_move": 0,
    "recompression": 0,
    "blur": 0,
}


def choose_text_box(annotation_path):
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = []

    for item in data.get("form", []):
        box = item.get("box")

        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = map(int, box)

        width = x2 - x1
        height = y2 - y1

        text = (item.get("text") or "").strip()

        if text and width >= 25 and height >= 12:
            candidates.append((box, text))

    if not candidates:
        return None

    # Prefer a reasonably sized text region.
    candidates.sort(
        key=lambda item: (
            abs((item[0][2] - item[0][0]) - 50),
            -len(item[1]),
        )
    )

    return candidates[0]


for image_path in files:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"SKIPPED unreadable: {image_path.name}")
        continue

    h, w = image.shape[:2]

    # ---------------------------------------------------------
    # 1. Authentic copy
    # ---------------------------------------------------------
    authentic_path = AUTHENTIC / f"{image_path.stem}.png"
    cv2.imwrite(str(authentic_path), image)
    created["authentic"] += 1

    # ---------------------------------------------------------
    # 2. Text edit
    # ---------------------------------------------------------
    annotation_path = SOURCE_ANNOTATIONS / f"{image_path.stem}.json"
    text_box = choose_text_box(annotation_path)

    if text_box:
        (x1, y1, x2, y2), original_text = text_box

        edited = image.copy()

        margin_x = max(2, int((x2 - x1) * 0.10))
        margin_y = max(2, int((y2 - y1) * 0.10))

        tx1 = x1 + margin_x
        ty1 = y1 + margin_y
        tx2 = x2 - margin_x
        ty2 = y2 - margin_y

        border = edited[
            max(0, y1 - 2):min(h, y2 + 2),
            max(0, x1 - 2):min(w, x2 + 2),
        ]

        background = np.median(
            border.reshape(-1, 3),
            axis=0
        ).astype(np.uint8)

        cv2.rectangle(
            edited,
            (tx1, ty1),
            (tx2, ty2),
            tuple(int(v) for v in background),
            thickness=-1,
        )

        cv2.putText(
            edited,
            "EDIT",
            (
                tx1,
                min(
                    ty2 - 2,
                    ty1 + max(10, ty2 - ty1 - 2),
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        output = TEXT / f"{image_path.stem}.png"
        cv2.imwrite(str(output), edited)
        created["text"] += 1

    # ---------------------------------------------------------
    # 3. Copy-move
    # ---------------------------------------------------------
    copy_move = image.copy()

    patch_w = int(w * 0.18)
    patch_h = int(h * 0.07)

    src_x = int(w * 0.12)
    src_y = int(h * 0.22)

    dst_x = int(w * 0.62)
    dst_y = int(h * 0.68)

    source_region = copy_move[
        src_y:src_y + patch_h,
        src_x:src_x + patch_w,
    ].copy()

    copy_move[
        dst_y:dst_y + patch_h,
        dst_x:dst_x + patch_w,
    ] = source_region

    output = COPY_MOVE / f"{image_path.stem}.png"
    cv2.imwrite(str(output), copy_move)
    created["copy_move"] += 1

    # ---------------------------------------------------------
    # 4. Localized recompression
    # ---------------------------------------------------------
    recompressed = image.copy()

    rx1 = int(w * 0.20)
    ry1 = int(h * 0.35)
    rx2 = int(w * 0.70)
    ry2 = int(h * 0.55)

    region = recompressed[ry1:ry2, rx1:rx2]

    encode_params = [
        cv2.IMWRITE_JPEG_QUALITY,
        25,
    ]

    success, encoded = cv2.imencode(
        ".jpg",
        region,
        encode_params,
    )

    if success:
        decoded = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

        if decoded is not None:
            decoded = cv2.resize(
                decoded,
                (rx2 - rx1, ry2 - ry1),
            )

            recompressed[
                ry1:ry2,
                rx1:rx2
            ] = decoded

    output = RECOMPRESSION / f"{image_path.stem}.png"
    cv2.imwrite(str(output), recompressed)
    created["recompression"] += 1

    # ---------------------------------------------------------
    # 5. Localized blur
    # ---------------------------------------------------------
    blurred = image.copy()

    bx1 = int(w * 0.25)
    by1 = int(h * 0.60)
    bx2 = int(w * 0.70)
    by2 = int(h * 0.75)

    region = blurred[by1:by2, bx1:bx2]

    blurred[
        by1:by2,
        bx1:bx2
    ] = cv2.GaussianBlur(
        region,
        (9, 9),
        0,
    )

    output = BLUR / f"{image_path.stem}.png"
    cv2.imwrite(str(output), blurred)
    created["blur"] += 1


print("=== FUNSD TRAINING/VALIDATION DATASET ===")
print(f"Source documents: {len(files)}")
print(f"Authentic: {created['authentic']}")
print(f"Text tampered: {created['text']}")
print(f"Copy-move tampered: {created['copy_move']}")
print(f"Recompression tampered: {created['recompression']}")
print(f"Blur tampered: {created['blur']}")
print(f"Total generated variants: {sum(created.values())}")
