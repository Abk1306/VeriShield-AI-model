def calculate_risk_score(
    tampered_probability: float,
    evidence: list[dict],
) -> dict:
    """
    Calculate a transparent 0-100 tampering risk score.

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

    # Base contribution from the ML model.
    score = tampered_probability * 70.0

    # Additional contribution from independent forensic evidence.
    severity_weights = {
        "LOW": 0.0,
        "MEDIUM": 8.0,
        "HIGH": 15.0,
    }

    evidence_contribution = 0.0

    for item in evidence:
        severity = item.get("severity", "LOW")
        evidence_contribution += severity_weights.get(
            severity,
            0.0,
        )

    score += min(
        evidence_contribution,
        30.0,
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
