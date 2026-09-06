import cv2
import numpy as np
from pathlib import Path


SOURCE = Path("dataset/authentic/test_document.png")
AUTHENTIC_DIR = Path("dataset/generated/authentic")
TAMPERED_DIR = Path("dataset/generated/tampered")

NUM_VARIATIONS = 50


def load_source():
    image = cv2.imread(str(SOURCE), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Unable to load source: {SOURCE}")

    return image


def text_edit(image, index):
    result = image.copy()

    height, width = result.shape[:2]

    x1 = 150
    y1 = 350
    x2 = min(width - 50, 800)
    y2 = min(height - 50, 440)

    cv2.rectangle(
        result,
        (x1, y1),
        (x2, y2),
        (245, 245, 245),
        -1,
    )

    cv2.putText(
        result,
        f"MODIFIED USER {index}",
        (x1 + 10, y1 + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (40, 40, 40),
        3,
        cv2.LINE_AA,
    )

    return result


def copy_move(image, index):
    result = image.copy()

    height, width = result.shape[:2]

    source_y1 = 270
    source_y2 = 340
    source_x1 = 160
    source_x2 = min(width - 100, 760)

    region = result[
        source_y1:source_y2,
        source_x1:source_x2,
    ].copy()

    target_y = min(height - region.shape[0] - 20, 520)
    target_x = 180

    result[
        target_y:target_y + region.shape[0],
        target_x:target_x + region.shape[1],
    ] = region

    return result


def region_recompression(image, index):
    result = image.copy()

    height, width = result.shape[:2]

    x1 = 150
    y1 = 350
    x2 = min(width - 50, 800)
    y2 = min(height - 50, 440)

    region = result[y1:y2, x1:x2]

    quality = 15 + (index % 5) * 10

    success, encoded = cv2.imencode(
        ".jpg",
        region,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )

    if not success:
        raise RuntimeError("JPEG encoding failed.")

    recompressed = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )

    result[y1:y2, x1:x2] = recompressed

    return result


def blur_edit(image, index):
    result = image.copy()

    height, width = result.shape[:2]

    x1 = 150
    y1 = 350
    x2 = min(width - 50, 800)
    y2 = min(height - 50, 440)

    region = result[y1:y2, x1:x2]

    kernel_size = 3 + (index % 3) * 2

    result[y1:y2, x1:x2] = cv2.GaussianBlur(
        region,
        (kernel_size, kernel_size),
        0,
    )

    return result


def create_authentic_variation(image, index):
    """
    Create visually equivalent authentic samples with
    small encoding variations. No intentional tampering.
    """
    encode_quality = 90 + (index % 10)

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, encode_quality],
    )

    if not success:
        raise RuntimeError("Unable to create authentic variation.")

    decoded = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )

    return decoded


def save_image(directory, filename, image):
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / filename

    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Unable to save {path}")


def main():
    image = load_source()

    AUTHENTIC_DIR.mkdir(parents=True, exist_ok=True)
    TAMPERED_DIR.mkdir(parents=True, exist_ok=True)

    # Generate authentic examples.
    for index in range(NUM_VARIATIONS):
        authentic = create_authentic_variation(image, index)

        save_image(
            AUTHENTIC_DIR,
            f"authentic_{index:03d}.jpg",
            authentic,
        )

    # Generate tampered examples.
    tamper_functions = [
        ("text", text_edit),
        ("copy_move", copy_move),
        ("recompression", region_recompression),
        ("blur", blur_edit),
    ]

    for name, function in tamper_functions:
        for index in range(NUM_VARIATIONS // len(tamper_functions)):
            tampered = function(image, index)

            save_image(
                TAMPERED_DIR,
                f"{name}_{index:03d}.jpg",
                tampered,
            )

    print("SYNTHETIC DATASET CREATED")
    print(f"Authentic directory: {AUTHENTIC_DIR}")
    print(f"Tampered directory:  {TAMPERED_DIR}")


if __name__ == "__main__":
    main()
