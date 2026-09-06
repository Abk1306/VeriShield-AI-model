from pathlib import Path

import cv2


SOURCE = Path("dataset/external/FUNSD_model_data/train/authentic")
OUTPUT = Path("dataset/external/FUNSD_model_data/train/authentic_augmented")

OUTPUT.mkdir(parents=True, exist_ok=True)

files = sorted(SOURCE.glob("*.png"))

created = 0

for path in files:
    image = cv2.imread(str(path))

    if image is None:
        continue

    # 1. Brightness
    bright = cv2.convertScaleAbs(
        image,
        alpha=1.0,
        beta=8,
    )
    cv2.imwrite(
        str(OUTPUT / f"{path.stem}_brightness.png"),
        bright,
    )
    created += 1

    # 2. Contrast
    contrast = cv2.convertScaleAbs(
        image,
        alpha=1.08,
        beta=0,
    )
    cv2.imwrite(
        str(OUTPUT / f"{path.stem}_contrast.png"),
        contrast,
    )
    created += 1

    # 3. JPEG recompression
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            55,
        ],
    )

    if success:
        jpeg_image = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

        if jpeg_image is not None:
            cv2.imwrite(
                str(OUTPUT / f"{path.stem}_jpeg.png"),
                jpeg_image,
            )
            created += 1

    # 4. Small rotation
    h, w = image.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        1.0,
        1.0,
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REPLICATE,
    )

    cv2.imwrite(
        str(OUTPUT / f"{path.stem}_rotation.png"),
        rotated,
    )
    created += 1


print("=== FUNSD AUTHENTIC AUGMENTATION ===")
print(f"Source authentic documents: {len(files)}")
print(f"Augmented authentic documents: {created}")
print(
    "Expected:",
    len(files) * 4,
)
