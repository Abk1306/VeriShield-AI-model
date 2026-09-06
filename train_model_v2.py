from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from verishield_ai.models.dataset_builder import build_feature_dataset


MODEL_PATH = Path(
    "verishield_ai/models/tampering_model_v2.joblib"
)


def collect_directory(directory):
    files = sorted(
        path
        for path in Path(directory).glob("*.png")
        if path.is_file()
    )

    return files


def extract_files(files, label):
    from verishield_ai.forensic.features import extract_features

    features = []
    labels = []

    for path in files:
        vector, _ = extract_features(str(path))
        features.append(vector)
        labels.append(label)

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def load_data():
    all_features = []
    all_labels = []

    # ---------------------------------------------------------
    # Existing synthetic authentic
    # ---------------------------------------------------------
    X, y, _ = build_feature_dataset(
        "dataset/generated/authentic",
        "dataset/generated/tampered",
    )

    synthetic_authentic = X[y == 0]
    synthetic_tampered = X[y == 1]

    all_features.extend([
        synthetic_authentic,
        synthetic_tampered,
    ])

    all_labels.extend([
        np.zeros(len(synthetic_authentic), dtype=np.int64),
        np.ones(len(synthetic_tampered), dtype=np.int64),
    ])

    # ---------------------------------------------------------
    # Existing benign authentic augmentation
    # ---------------------------------------------------------
    augmented_files = sorted(
        Path("dataset/generated/authentic_augmented").glob("*.jpg")
    )

    X_aug, y_aug = extract_files(
        augmented_files,
        0,
    )

    all_features.append(X_aug)
    all_labels.append(y_aug)

    # ---------------------------------------------------------
    # FUNSD training data
    # ---------------------------------------------------------
    root = Path("dataset/external/FUNSD_model_data/train")

    authentic_files = collect_directory(
        root / "authentic"
    )

    tampered_files = []

    for category in [
        "tampered_text",
        "tampered_copy_move",
        "tampered_recompression",
        "tampered_blur",
    ]:
        tampered_files.extend(
            collect_directory(root / category)
        )

    X_funsd_auth, y_funsd_auth = extract_files(
        authentic_files,
        0,
    )

    X_funsd_tamp, y_funsd_tamp = extract_files(
        tampered_files,
        1,
    )

    all_features.extend([
        X_funsd_auth,
        X_funsd_tamp,
    ])

    all_labels.extend([
        y_funsd_auth,
        y_funsd_tamp,
    ])

    X_final = np.concatenate(
        all_features,
        axis=0,
    )

    y_final = np.concatenate(
        all_labels,
        axis=0,
    )

    return X_final, y_final


def train():
    X, y = load_data()

    print("=== VERISHIELD MODEL V2 ===")
    print(f"Total training samples: {len(X)}")
    print(f"Authentic samples: {(y == 0).sum()}")
    print(f"Tampered samples: {(y == 1).sum()}")
    print(f"Feature count: {X.shape[1]}")

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

    model.fit(X, y)

    train_predictions = model.predict(X)

    print(
        f"\nTraining accuracy: "
        f"{accuracy_score(y, train_predictions):.4f}"
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, MODEL_PATH)

    print(f"\nMODEL SAVED: {MODEL_PATH}")


if __name__ == "__main__":
    train()
