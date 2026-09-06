from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from verishield_ai.forensic.features import extract_features


MODEL_PATH = Path("verishield_ai/models/tampering_model_v4.joblib")


def collect_features(directory):
    directory = Path(directory)
    rows = []

    files = sorted(
        p for p in directory.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )

    for i, path in enumerate(files, 1):
        features, _ = extract_features(str(path))
        rows.append(features)

        if i % 25 == 0 or i == len(files):
            print(f"  {directory}: {i}/{len(files)}")

    if not rows:
        raise ValueError(f"No images found in {directory}")

    return np.asarray(rows, dtype=np.float32)


def add_dataset(X_parts, y_parts, directory, label):
    X = collect_features(directory)

    X_parts.append(X)
    y_parts.append(
        np.full(len(X), label, dtype=np.int64)
    )


def load_data():
    X_parts = []
    y_parts = []

    print("\n=== BUILDING V4 DATASET ===")

    # Synthetic authentic
    add_dataset(
        X_parts,
        y_parts,
        "dataset/generated/authentic",
        0,
    )

    # Synthetic benign authentic
    add_dataset(
        X_parts,
        y_parts,
        "dataset/generated/authentic_augmented",
        0,
    )

    # Synthetic tampered
    add_dataset(
        X_parts,
        y_parts,
        "dataset/generated/tampered",
        1,
    )

    # FUNSD authentic
    add_dataset(
        X_parts,
        y_parts,
        "dataset/external/FUNSD_model_data/train/authentic",
        0,
    )

    # FUNSD benign authentic
    add_dataset(
        X_parts,
        y_parts,
        "dataset/external/FUNSD_model_data/train/authentic_augmented",
        0,
    )

    # FUNSD tampered categories
    for category in [
        "tampered_text",
        "tampered_copy_move",
        "tampered_recompression",
        "tampered_blur",
    ]:
        add_dataset(
            X_parts,
            y_parts,
            f"dataset/external/FUNSD_model_data/train/{category}",
            1,
        )

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    return X, y


def train():
    X, y = load_data()

    print("\n=== V4 DATASET SUMMARY ===")
    print(f"Total samples: {len(X)}")
    print(f"Authentic samples: {int((y == 0).sum())}")
    print(f"Tampered samples: {int((y == 1).sum())}")
    print(f"Feature count: {X.shape[1]}")

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
                    n_estimators=400,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    print("\nTraining V4...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n=== VERISHIELD MODEL V4 ===")
    print(f"Training split: {len(X_train)}")
    print(f"Test split: {len(X_test)}")
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
    train()
