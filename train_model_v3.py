from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from verishield_ai.models.dataset_builder import build_feature_dataset


MODEL_PATH = Path("verishield_ai/models/tampering_model_v3.joblib")


def load_training_data():
    synthetic_authentic = Path("dataset/generated/authentic")
    synthetic_augmented = Path("dataset/generated/authentic_augmented")
    synthetic_tampered = Path("dataset/generated/tampered")

    funsd_authentic = Path("dataset/external/FUNSD_model_data/train/authentic")
    funsd_augmented = Path(
        "dataset/external/FUNSD_model_data/train/authentic_augmented"
    )
    funsd_tampered = Path("dataset/external/FUNSD_model_data/train")

    X_syn, y_syn, _ = build_feature_dataset(
        str(synthetic_authentic),
        str(synthetic_tampered),
    )

    X_syn_aug, y_syn_aug, _ = build_feature_dataset(
        str(synthetic_augmented),
        str(synthetic_tampered),
    )

    X_funsd_text, y_funsd_text, _ = build_feature_dataset(
        str(funsd_authentic),
        str(funsd_tampered / "tampered_text"),
    )

    X_funsd_copy, y_funsd_copy, _ = build_feature_dataset(
        str(funsd_authentic),
        str(funsd_tampered / "tampered_copy_move"),
    )

    X_funsd_recomp, y_funsd_recomp, _ = build_feature_dataset(
        str(funsd_authentic),
        str(funsd_tampered / "tampered_recompression"),
    )

    X_funsd_blur, y_funsd_blur, _ = build_feature_dataset(
        str(funsd_authentic),
        str(funsd_tampered / "tampered_blur"),
    )

    synthetic_auth = X_syn[y_syn == 0]
    synthetic_tampered = X_syn[y_syn == 1]
    synthetic_aug_auth = X_syn_aug[y_syn_aug == 0]

    funsd_auth = X_funsd_text[y_funsd_text == 0]
    funsd_tampered = np.concatenate(
        [
            X_funsd_text[y_funsd_text == 1],
            X_funsd_copy[y_funsd_copy == 1],
            X_funsd_recomp[y_funsd_recomp == 1],
            X_funsd_blur[y_funsd_blur == 1],
        ],
        axis=0,
    )

    X = np.concatenate(
        [
            synthetic_auth,
            synthetic_aug_auth,
            synthetic_tampered,
            funsd_auth,
            funsd_tampered,
        ],
        axis=0,
    )

    y = np.concatenate(
        [
            np.zeros(len(synthetic_auth), dtype=np.int64),
            np.zeros(len(synthetic_aug_auth), dtype=np.int64),
            np.ones(len(synthetic_tampered), dtype=np.int64),
            np.zeros(len(funsd_auth), dtype=np.int64),
            np.ones(len(funsd_tampered), dtype=np.int64),
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
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n=== VERISHIELD MODEL V3 ===")
    print(f"Total training samples: {len(X)}")
    print(f"Authentic samples: {int((y == 0).sum())}")
    print(f"Tampered samples: {int((y == 1).sum())}")
    print(f"Training split: {len(X_train)}")
    print(f"Test split: {len(X_test)}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

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
