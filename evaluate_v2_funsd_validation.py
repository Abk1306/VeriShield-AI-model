from pathlib import Path

from verishield_ai.models.inference import predict_document


MODEL = "verishield_ai/models/tampering_model_v2.joblib"
ROOT = Path("dataset/external/FUNSD_model_data/validation")

categories = {
    "authentic": 0,
    "tampered_text": 1,
    "tampered_copy_move": 1,
    "tampered_recompression": 1,
    "tampered_blur": 1,
}

correct = 0
total = 0

print("=== VERISHIELD V2 FUNSD VALIDATION ===")

for category, true_label in categories.items():
    directory = ROOT / category
    files = sorted(directory.glob("*.png"))

    category_correct = 0

    for path in files:
        result = predict_document(
            str(path),
            model_path=MODEL,
        )

        predicted_label = (
            1 if result["classification"] == "TAMPERED" else 0
        )

        if predicted_label == true_label:
            category_correct += 1

        total += 1

    correct += category_correct

    print(
        f"{category}: "
        f"{category_correct}/{len(files)} correct"
    )

print("\n=== SUMMARY ===")
print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Incorrect: {total - correct}")
print(f"Accuracy: {correct / total:.2%}")
