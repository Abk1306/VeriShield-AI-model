from pathlib import Path

import joblib
import numpy as np

from verishield_ai.forensic.features import extract_features


DEFAULT_MODEL_PATH = Path(
    "verishield_ai/models/tampering_model.joblib"
)


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """
    Load the trained tampering classifier.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(model_path)


def predict_document(
    file_path: str,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict:
    """
    Predict whether a document shows signs of tampering.

    Returns:
        Dictionary containing classification and probabilities.
    """
    model = load_model(model_path)

    features, feature_names = extract_features(
        file_path
    )

    probabilities = model.predict_proba(
        features.reshape(1, -1)
    )[0]

    prediction = int(
        model.predict(
            features.reshape(1, -1)
        )[0]
    )

    authentic_probability = float(
        probabilities[0]
    )

    tampered_probability = float(
        probabilities[1]
    )

    classification = (
        "TAMPERED"
        if prediction == 1
        else "AUTHENTIC"
    )

    confidence = max(
        authentic_probability,
        tampered_probability,
    )

    return {
        "classification": classification,
        "confidence": confidence,
        "tampered_probability": tampered_probability,
        "authentic_probability": authentic_probability,
        "feature_count": len(feature_names),
    }
