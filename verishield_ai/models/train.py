from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from verishield_ai.models.dataset_builder import build_feature_dataset


MODEL_PATH = Path("verishield_ai/models/tampering_model.joblib")


def load_training_data():
    """
    Build the combined training dataset.

    Authentic:
        - original authentic samples
        - benign augmented authentic samples

    Tampered:
        - generated tampered samples
    """
    authentic_original = Path("dataset/generated/authentic")
    authentic_augmented = Path("dataset/generated/authentic_augmented")
    tampered_dir = Path("dataset/generated/tampered")

    X_original, y_original, _ = build_feature_dataset(
        str(authentic_original),
        str(tampered_dir),
    )

    X_augmented, y_augmented, _ = build_feature_dataset(
        str(authentic_augmented),
        str(tampered_dir),
    )

    original_authentic = X_original[y_original == 0]
    augmented_authentic = X_augmented[y_augmented == 0]
    tampered_samples = X_original[y_original == 1]

    X = np.concatenate(
        [
            original_authentic,
            augmented_authentic,
            tampered_samples,
        ],
        axis=0,
    )

    y = np.concatenate(
        [
            np.zeros(len(original_authentic), dtype=np.int64),
            np.zeros(len(augmented_authentic), dtype=np.int64),
            np.ones(len(tampered_samples), dtype=np.int64),
        ],
        axis=0,
    )

    return X, y


def train_model():
    X, y = load_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n=== VERISHIELD ROBUST MODEL ===")
    print(f"Total samples: {len(X)}")
    print(f"Authentic samples: {int((y == 0).sum())}")
    print(f"Tampered samples: {int((y == 1).sum())}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["AUTHENTIC", "TAMPERED"],
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nMODEL SAVED: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
