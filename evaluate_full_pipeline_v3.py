from pathlib import Path

from verishield_ai.models.inference import predict_document
from verishield_ai.forensic.ela import compute_ela
from verishield_ai.forensic.noise import compute_noise_map
from verishield_ai.forensic.compression import compute_compression_map
from verishield_ai.evidence import build_evidence
from verishield_ai.scoring import calculate_risk_score
from verishield_ai.preprocessing import preprocess_document


BASE = Path("dataset/external/FUNSD_final_test")
MODEL = Path("verishield_ai/models/tampering_model_v3.joblib")

CATEGORIES = {
    "authentic": "AUTHENTIC",
    "tampered_text": "TAMPERED",
    "tampered_copy_move": "TAMPERED",
    "tampered_recompression": "TAMPERED",
    "tampered_blur": "TAMPERED",
}


def analyze_with_v3(path):
    image = preprocess_document(path)

    ela_map = compute_ela(image)
    noise_map = compute_noise_map(image)
    compression_map = compute_compression_map(image)

    prediction = predict_document(
        str(path),
        model_path=MODEL,
    )

    evidence = build_evidence(
        ela_map=ela_map,
        noise_map=noise_map,
        compression_map=compression_map,
        tampered_probability=prediction["tampered_probability"],
    )

    risk = calculate_risk_score(
        tampered_probability=prediction["tampered_probability"],
        evidence=evidence,
    )

    return risk["classification"], risk["tampering_score"]


total = 0
correct = 0

print("\n=== VERISHIELD FULL PIPELINE V3 FINAL TEST ===")
print(f"Model: {MODEL}")
print("Samples: 50\n")

for category, expected in CATEGORIES.items():

    directory = BASE / category
    files = sorted(directory.glob("*"))

    category_correct = 0

    for path in files:
        classification, score = analyze_with_v3(path)

        total += 1

        if classification == expected:
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

print("\nStatus: V3 FORENSIC + ML PIPELINE TEST COMPLETE")
