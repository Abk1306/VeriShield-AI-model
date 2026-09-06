from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from verishield_ai.models.inference import predict_document


def evaluate_dataset(authentic_dir, tampered_dir):
    authentic_files = sorted(Path(authentic_dir).glob("*.jpg"))
    tampered_files = sorted(Path(tampered_dir).glob("*.jpg"))

    paths = authentic_files + tampered_files
    true_labels = [0] * len(authentic_files) + [1] * len(tampered_files)

    predictions = []

    for path in paths:
        result = predict_document(str(path))
        predictions.append(1 if result["classification"] == "TAMPERED" else 0)

    print("\n=== VERISHIELD DATASET EVALUATION ===")
    print(f"Total samples: {len(paths)}")
    print(f"Authentic samples: {len(authentic_files)}")
    print(f"Tampered samples: {len(tampered_files)}")
    print(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            predictions,
            target_names=["AUTHENTIC", "TAMPERED"],
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, predictions))


if __name__ == "__main__":
    evaluate_dataset(
        "dataset/generated/authentic",
        "dataset/generated/tampered",
    )
