from pathlib import Path

from verishield_ai.pipeline import analyze_document


BASE = Path("dataset/external/FUNSD_final_test")

CATEGORIES = {
    "authentic": "AUTHENTIC",
    "tampered_text": "TAMPERED",
    "tampered_copy_move": "TAMPERED",
    "tampered_recompression": "TAMPERED",
    "tampered_blur": "TAMPERED",
}

total = 0
correct = 0

print("\n=== VERISHIELD FULL PIPELINE FINAL TEST ===")
print("Model: V3")
print("Samples: 50")
print()

for category, expected in CATEGORIES.items():

    directory = BASE / category
    files = sorted(directory.glob("*"))

    category_correct = 0

    for path in files:
        result = analyze_document(str(path))

        prediction = result["classification"]

        total += 1

        if prediction == expected:
            correct += 1
            category_correct += 1

    print(
        f"{category}: "
        f"{category_correct}/{len(files)} correct"
    )

print("\n=== FINAL SUMMARY ===")
print(f"Total samples: {total}")
print(f"Correct: {correct}")
print(f"Incorrect: {total - correct}")

if total:
    print(f"Accuracy: {correct / total:.4f}")

print("\nStatus: COMPLETE PIPELINE TEST FINISHED")
