def calculate_risk_score(
    tampered_probability: float,
    evidence: list[dict],
) -> dict:
    """
    Calculate a transparent 0-100 tampering risk score.

    The ML prediction is the primary signal. Forensic evidence
    provides supporting risk rather than independently dominating
    the final classification.

    This is a development scoring model. It is not a calibrated
    probability of forgery and must be validated against a larger,
    representative dataset before real-world deployment.
    """
    if not 0.0 <= tampered_probability <= 1.0:
        raise ValueError(
            "tampered_probability must be between 0 and 1."
        )

    if not isinstance(evidence, list):
        raise ValueError("evidence must be a list.")

    # Primary contribution from the ML model.
    score = tampered_probability * 85.0

    # Forensic evidence provides supporting information.
    # Multiple forensic signals are deliberately capped so that
    # normal image processing/compression cannot overwhelm the
    # model prediction.
    severity_weights = {
        "LOW": 0.0,
        "MEDIUM": 3.0,
        "HIGH": 5.0,
    }

    evidence_contribution = 0.0

    for item in evidence:
        evidence_contribution += severity_weights.get(
            item.get("severity", "LOW"),
            0.0,
        )

    score += min(
        evidence_contribution,
        15.0,
    )

    score = max(
        0.0,
        min(score, 100.0),
    )

    if score >= 70:
        classification = "TAMPERED"
    elif score >= 30:
        classification = "SUSPICIOUS"
    else:
        classification = "AUTHENTIC"

    return {
        "tampering_score": round(score, 2),
        "classification": classification,
    }
