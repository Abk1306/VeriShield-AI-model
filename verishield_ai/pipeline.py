from pathlib import Path

from verishield_ai.preprocessing import preprocess_document
from verishield_ai.forensic.ela import compute_ela
from verishield_ai.forensic.noise import compute_noise_map
from verishield_ai.forensic.compression import compute_compression_map
from verishield_ai.models.inference import predict_document
from verishield_ai.evidence import build_evidence
from verishield_ai.scoring import calculate_risk_score
from verishield_ai.localization import (
    create_anomaly_map,
    detect_suspicious_regions,
)


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def analyze_document(file_path: str | Path) -> dict:
    """
    Analyze a document for signs of digital tampering.

    This is the main public VeriShield API.

    Parameters
    ----------
    file_path:
        Path to a supported document image.

    Returns
    -------
    dict
        Structured tampering-analysis result.

    Notes
    -----
    The result estimates signs of digital manipulation.
    It does not establish official document authenticity.
    """

    file_path = Path(file_path)

    # ---------------------------------------------------------
    # 1. Validate input
    # ---------------------------------------------------------
    if not file_path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Document path is not a file: {file_path}"
        )

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document type: {file_path.suffix}. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # ---------------------------------------------------------
    # 2. Preprocess
    # ---------------------------------------------------------
    image = preprocess_document(file_path)

    # ---------------------------------------------------------
    # 3. Forensic analysis
    # ---------------------------------------------------------
    ela_map = compute_ela(image)
    noise_map = compute_noise_map(image)
    compression_map = compute_compression_map(image)

    # ---------------------------------------------------------
    # 4. ML prediction
    # ---------------------------------------------------------
    prediction = predict_document(file_path)

    # ---------------------------------------------------------
    # 5. Evidence aggregation
    # ---------------------------------------------------------
    evidence = build_evidence(
        ela_map=ela_map,
        noise_map=noise_map,
        compression_map=compression_map,
        tampered_probability=prediction[
            "tampered_probability"
        ],
    )

    # ---------------------------------------------------------
    # 6. Risk scoring
    # ---------------------------------------------------------
    risk = calculate_risk_score(
        tampered_probability=prediction[
            "tampered_probability"
        ],
        evidence=evidence,
    )

    # ---------------------------------------------------------
    # 7. Suspicious-region localization
    # ---------------------------------------------------------
    anomaly_map = create_anomaly_map(
        ela_map=ela_map,
        noise_map=noise_map,
        compression_map=compression_map,
    )

    regions = detect_suspicious_regions(
        anomaly_map,
        threshold=80,
        min_area=200,
        max_area_ratio=0.20,
        max_regions=10,
    )

    # ---------------------------------------------------------
    # 8. Final structured result
    # ---------------------------------------------------------
    return {
        "classification": risk["classification"],
        "tampering_score": risk["tampering_score"],
        "confidence": prediction["confidence"],
        "tampered_probability": prediction[
            "tampered_probability"
        ],
        "authentic_probability": prediction[
            "authentic_probability"
        ],
        "evidence": evidence,
        "regions": regions,
        "feature_count": prediction["feature_count"],
    }
