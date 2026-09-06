from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_document(file_path: str) -> np.ndarray:
    """
    Load a document image and return it as an OpenCV BGR image.

    Args:
        file_path: Path to the document image.

    Returns:
        Image as a NumPy array in BGR format.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is unsupported or cannot be decoded.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document format: {path.suffix}. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Unable to decode document image: {file_path}")

    return image


def normalize_image(
    image: np.ndarray,
    max_dimension: int = 2000,
) -> np.ndarray:
    """
    Normalize an image while preserving its aspect ratio.

    Images larger than max_dimension on their longest side are resized.
    Smaller images are left unchanged.

    Args:
        image: OpenCV BGR image.
        max_dimension: Maximum allowed width or height.

    Returns:
        Normalized OpenCV BGR image.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image.")

    if max_dimension <= 0:
        raise ValueError("max_dimension must be greater than zero.")

    height, width = image.shape[:2]
    longest_side = max(height, width)

    if longest_side <= max_dimension:
        return image.copy()

    scale = max_dimension / longest_side

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def preprocess_document(
    file_path: str,
    max_dimension: int = 2000,
) -> np.ndarray:
    """
    Load and normalize a document image.

    This is the main preprocessing entry point used by the pipeline.
    """
    image = load_document(file_path)
    return normalize_image(image, max_dimension=max_dimension)
