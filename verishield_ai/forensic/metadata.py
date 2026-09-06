from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS


def analyze_metadata(file_path: str) -> dict:
    """
    Extract available image metadata for forensic analysis.

    Metadata is treated as a supporting signal and should not be
    interpreted as proof of document tampering by itself.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    try:
        image = Image.open(path)
    except Exception as exc:
        raise ValueError(
            f"Unable to read image metadata: {file_path}"
        ) from exc

    metadata = {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "format": image.format,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "has_exif": False,
        "exif": {},
    }

    exif_data = image.getexif()

    if exif_data:
        metadata["has_exif"] = True

        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, str(tag_id))

            try:
                if isinstance(value, bytes):
                    value = value.decode(
                        "utf-8",
                        errors="replace",
                    )
                else:
                    value = str(value)

                metadata["exif"][tag_name] = value
            except Exception:
                metadata["exif"][tag_name] = "<unreadable>"

    return metadata


def metadata_statistics(metadata: dict) -> dict:
    """
    Convert metadata into simple forensic indicators.
    """
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary.")

    exif = metadata.get("exif", {})

    software = exif.get("Software")
    make = exif.get("Make")
    model = exif.get("Model")

    return {
        "format": metadata.get("format"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "has_exif": bool(metadata.get("has_exif")),
        "has_software_tag": software is not None,
        "has_camera_make": make is not None,
        "has_camera_model": model is not None,
    }
