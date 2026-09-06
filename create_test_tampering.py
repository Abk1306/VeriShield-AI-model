import cv2
import numpy as np
from pathlib import Path


SOURCE = Path("dataset/authentic/test_document.png")
OUTPUT_DIR = Path("dataset/tampered")


def load_source():
    image = cv2.imread(str(SOURCE), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Unable to load {SOURCE}")

    return image


def create_text_edit(image):
    result = image.copy()

    cv2.rectangle(
        result,
        (170, 365),
        (760, 435),
        (245, 245, 245),
        -1,
    )

    cv2.putText(
        result,
        "Name: MODIFIED USER",
        (180, 415),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (40, 40, 40),
        3,
        cv2.LINE_AA,
    )

    return result


def create_copy_move(image):
    result = image.copy()

    source_region = result[280:350, 170:760].copy()

    target_y = 550
    target_x = 170

    result[
        target_y:target_y + source_region.shape[0],
        target_x:target_x + source_region.shape[1]
    ] = source_region

    return result


def create_region_recompression(image):
    result = image.copy()

    x1, y1 = 170, 365
    x2, y2 = 760, 435

    region = result[y1:y2, x1:x2]

    success, encoded = cv2.imencode(
        ".jpg",
        region,
        [cv2.IMWRITE_JPEG_QUALITY, 25],
    )

    if not success:
        raise RuntimeError("JPEG encoding failed.")

    recompressed = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )

    result[y1:y2, x1:x2] = recompressed

    return result


def save(name, image):
    path = OUTPUT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Unable to save {path}")

    print(f"CREATED: {path}")


def main():
    image = load_source()

    save(
        "tampered_text_edit.png",
        create_text_edit(image),
    )

    save(
        "tampered_copy_move.png",
        create_copy_move(image),
    )

    save(
        "tampered_recompression.png",
        create_region_recompression(image),
    )

    print("\nALL CONTROLLED TAMPERED SAMPLES CREATED")


if __name__ == "__main__":
    main()
