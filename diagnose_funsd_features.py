from pathlib import Path

from verishield_ai.forensic.features import extract_features


original_root = Path("dataset/external/FUNSD/dataset/testing_data/images")
tampered_root = Path("dataset/external/FUNSD_eval/tampered")

pairs = []

for tampered in sorted(tampered_root.glob("*.png")):
    original_name = tampered.name.replace("_copy_move", "")
    original = original_root / original_name

    if original.exists():
        original_features, _ = extract_features(str(original))
        tampered_features, _ = extract_features(str(tampered))

        difference = abs(tampered_features - original_features)

        pairs.append(
            (
                tampered.name,
                difference.mean(),
                difference.max(),
                difference.sum(),
            )
        )

print("=== FUNSD FEATURE SHIFT DIAGNOSTIC ===")
print(f"Pairs compared: {len(pairs)}")

for name, mean_diff, max_diff, sum_diff in pairs:
    print(
        f"{name}: "
        f"mean_feature_shift={mean_diff:.4f}, "
        f"max_feature_shift={max_diff:.4f}, "
        f"total_shift={sum_diff:.4f}"
    )
