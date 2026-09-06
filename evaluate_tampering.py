from pathlib import Path

from verishield_ai.models.inference import predict_document

root = Path("dataset/generated/tampered")

for prefix in ["blur_", "copy_move_", "recompression_", "text_"]:
    files = sorted(root.glob(prefix + "*.jpg"))
    results = [predict_document(str(path)) for path in files]

    detected = sum(
        result["classification"] == "TAMPERED"
        for result in results
    )

    probabilities = [
        result["tampered_probability"]
        for result in results
    ]

    print(
        f"{prefix[:-1]}: "
        f"{detected}/{len(results)} detected | "
        f"min probability={min(probabilities):.3f} | "
        f"max probability={max(probabilities):.3f}"
    )
