from pathlib import Path

from verishield_ai.models.inference import predict_document

root = Path("dataset/external/FUNSD/dataset/testing_data/images")
files = sorted(root.glob("*.png"))

results = [predict_document(str(path)) for path in files]

false_positives = [
    (path.name, result["tampered_probability"])
    for path, result in zip(files, results)
    if result["classification"] == "TAMPERED"
]

print("=== FUNSD EXTERNAL AUTHENTIC TEST ===")
print(f"Total documents: {len(results)}")
print(f"AUTHENTIC predictions: {len(results) - len(false_positives)}")
print(f"TAMPERED predictions: {len(false_positives)}")
print(
    f"False-positive rate: "
    f"{len(false_positives) / len(results):.2%}"
)

if false_positives:
    print("\nDocuments classified as TAMPERED:")
    for name, probability in false_positives:
        print(f"{name}: tampered_probability={probability:.3f}")

print(
    "\nMaximum tampered probability: "
    f"{max(r['tampered_probability'] for r in results):.3f}"
)
print(
    "Minimum confidence: "
    f"{min(r['confidence'] for r in results):.3f}"
)
