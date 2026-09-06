from verishield_ai.forensic.ela import ela_statistics
from verishield_ai.forensic.noise import noise_statistics
from verishield_ai.forensic.compression import compression_statistics


def _severity(value: float, low: float, high: float) -> str:
    """
    Convert a normalized signal into a simple evidence severity.
    """
    if value >= high:
        return "HIGH"

    if value >= low:
        return "MEDIUM"

    return "LOW"


def build_evidence(
    ela_map,
    noise_map,
    compression_map,
    tampered_probability: float,
) -> list[dict]:
    """
    Build explainable forensic evidence from detector outputs.

    Thresholds here are intentionally conservative baseline values.
    They are not scientifically calibrated forensic thresholds.
    """
    ela = ela_statistics(ela_map)
    noise = noise_statistics(noise_map)
    compression = compression_statistics(compression_map)

    evidence = []

    ela_severity = _severity(
        ela["mean"],
        low=2.0,
        high=3.0,
    )

    if ela_severity != "LOW":
        evidence.append(
            {
                "type": "ela_anomaly",
                "severity": ela_severity,
                "value": ela["mean"],
                "description": (
                    "Elevated JPEG recompression error was "
                    "observed in the document."
                ),
            }
        )

    noise_severity = _severity(
        noise["high_noise_ratio"],
        low=0.0002,
        high=0.0005,
    )

    if noise_severity != "LOW":
        evidence.append(
            {
                "type": "noise_anomaly",
                "severity": noise_severity,
                "value": noise["high_noise_ratio"],
                "description": (
                    "Localized high-frequency noise "
                    "inconsistency was observed."
                ),
            }
        )

    compression_severity = _severity(
        compression["high_artifact_ratio"],
        low=0.0004,
        high=0.001,
    )

    if compression_severity != "LOW":
        evidence.append(
            {
                "type": "compression_anomaly",
                "severity": compression_severity,
                "value": compression["high_artifact_ratio"],
                "description": (
                    "Potential compression artifact "
                    "inconsistency was observed."
                ),
            }
        )

    if tampered_probability >= 0.70:
        evidence.append(
            {
                "type": "ml_prediction",
                "severity": "HIGH",
                "value": tampered_probability,
                "description": (
                    "The machine-learning model indicates "
                    "a high probability of tampering."
                ),
            }
        )
    elif tampered_probability >= 0.40:
        evidence.append(
            {
                "type": "ml_prediction",
                "severity": "MEDIUM",
                "value": tampered_probability,
                "description": (
                    "The machine-learning model indicates "
                    "an elevated probability of tampering."
                ),
            }
        )

    return evidence
