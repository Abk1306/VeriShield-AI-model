from pathlib import Path

from verishield_ai.models.inference import predict_document


root = Path("dataset/external/FUNSD_eval/tampered")
files = sorted(root.glob("*.png"))

results = []

for path in files:
    result = predict_document(str(path))
    results.append(result)

detected = sum(
    result["classification"] == "TAMPERED"
    for result in results
)

print("=== FUNSD TAMPERED EXTERNAL TEST ===")
print(f"Total: {len(results)}")
print(f"Detected TAMPERED: {detected}")
print(f"Missed: {len(results) - detected}")

print("\nIndividual results:")

for path, result in zip(files, results):
    print(
        f"{path.name}: "
        f"{result['classification']}, "
        f"probability={result['tampered_probability']:.3f}, "
        f"confidence={result['confidence']:.3f}"
    )
