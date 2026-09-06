from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from verishield_ai.forensic.features import extract_features


MODEL_PATH = Path("verishield_ai/models/tampering_model_v4.joblib")
BASE = Path("dataset/external/FUNSD_final_test")

CATEGORIES = {
    "authentic": 0,
    "tampered_text": 1,
    "tampered_copy_move": 1,
    "tampered_recompression": 1,
    "tampered_blur": 1,
}


def main():
    model = joblib.load(MODEL_PATH)

    predictions = []
    true_labels = []

    print("\n=== VERISHIELD V3 FINAL FUNSD TEST ===")
    print(f"Model: {MODEL_PATH}")
    print("Final-test documents: 10")
    print("Total evaluation samples: 50\n")

    for category, true_label in CATEGORIES.items():
        directory = BASE / category
        files = sorted(directory.glob("*"))

        if not directory.exists():
            print(f"ERROR: Missing directory: {directory}")
            return

        correct = 0

        for path in files:
            features, _ = extract_features(str(path))
            probability = model.predict_proba([features])[0][1]
            prediction = 1 if probability >= 0.5 else 0

            predictions.append(prediction)
            true_labels.append(true_label)

            if prediction == true_label:
                correct += 1

        print(f"{category}: {correct}/{len(files)} correct")

    correct_total = sum(
        prediction == label
        for prediction, label in zip(predictions, true_labels)
    )

    accuracy = accuracy_score(true_labels, predictions)

    print("\n=== FINAL SUMMARY ===")
    print(f"Total samples: {len(predictions)}")
    print(f"Correct: {correct_total}")
    print(f"Incorrect: {len(predictions) - correct_total}")
    print(f"Accuracy: {accuracy:.4f}")

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
    main()
