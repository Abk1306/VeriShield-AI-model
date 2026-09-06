from pathlib import Path

import cv2


SOURCE = Path("dataset/external/FUNSD/dataset/testing_data/images")
OUTPUT = Path("dataset/external/FUNSD_final_test")

FINAL_FILES = [
    "87147607.png",
    "87332450.png",
    "87428306.png",
    "87528321.png",
    "87528380.png",
    "87594142_87594144.png",
    "89856243.png",
    "91814768_91814769.png",
    "92380595.png",
    "93106788.png",
]

CATEGORIES = [
    "authentic",
    "tampered_text",
    "tampered_copy_move",
    "tampered_recompression",
    "tampered_blur",
]

for category in CATEGORIES:
    (OUTPUT / category).mkdir(parents=True, exist_ok=True)


for filename in FINAL_FILES:
    source_path = SOURCE / filename
    image = cv2.imread(str(source_path))

    if image is None:
        print(f"ERROR: Could not read {filename}")
        continue

    stem = Path(filename).stem

    # 1. Authentic
    cv2.imwrite(
        str(OUTPUT / "authentic" / filename),
        image,
    )

    # 2. Text-region modification
    text_edit = image.copy()
    h, w = text_edit.shape[:2]

    x1 = int(w * 0.25)
    y1 = int(h * 0.25)
    x2 = int(w * 0.55)
    y2 = int(h * 0.30)

    cv2.rectangle(
        text_edit,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        -1,
    )

    cv2.putText(
        text_edit,
        "EDITED",
        (x1 + 5, y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(OUTPUT / "tampered_text" / f"{stem}.jpg"),
        text_edit,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    # 3. Copy-move
    copy_move = image.copy()

    source_x = int(w * 0.10)
    source_y = int(h * 0.25)
    patch_w = max(20, int(w * 0.20))
    patch_h = max(20, int(h * 0.08))

    target_x = int(w * 0.60)
    target_y = int(h * 0.65)

    source_patch = image[
        source_y:source_y + patch_h,
        source_x:source_x + patch_w,
    ].copy()

    target_x2 = min(w, target_x + patch_w)
    target_y2 = min(h, target_y + patch_h)

    actual_w = target_x2 - target_x
    actual_h = target_y2 - target_y

    if actual_w > 0 and actual_h > 0:
        source_patch = cv2.resize(
            source_patch,
            (actual_w, actual_h),
        )

        copy_move[
            target_y:target_y2,
            target_x:target_x2,
        ] = source_patch

    cv2.imwrite(
        str(OUTPUT / "tampered_copy_move" / f"{stem}.jpg"),
        copy_move,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    # 4. Recompression
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 25],
    )

    if success:
        recompressed = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

        cv2.imwrite(
            str(OUTPUT / "tampered_recompression" / f"{stem}.jpg"),
            recompressed,
            [cv2.IMWRITE_JPEG_QUALITY, 25],
        )

    # 5. Blur region
    blurred = image.copy()

    bx1 = int(w * 0.30)
    by1 = int(h * 0.40)
    bx2 = min(w, int(w * 0.60))
    by2 = min(h, int(h * 0.50))

    region = blurred[by1:by2, bx1:bx2]

    if region.size > 0:
        blurred[by1:by2, bx1:bx2] = cv2.GaussianBlur(
            region,
            (21, 21),
            0,
        )

    cv2.imwrite(
        str(OUTPUT / "tampered_blur" / f"{stem}.jpg"),
        blurred,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )


print("=== FINAL FUNSD TEST DATA CREATED ===")

for category in CATEGORIES:
    count = len(list((OUTPUT / category).glob("*")))
    print(f"{category}: {count}")

print("Expected: 10 per category")
print("Total expected: 50")
