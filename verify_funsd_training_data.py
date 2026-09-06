from pathlib import Path


ROOT = Path("dataset/external/FUNSD_train_eval")

final_test_names = {
    "87147607.png",
    "87332450.png",
    "87428306.png",
    "87528321.png",
    "87528380.png",
    "87594142_87594144.png",
    "89856243.png",
    "91814768_91814769.png",
    "92380595.png",
    "93106788.png",
}

directories = {
    "authentic": ROOT / "authentic",
    "text": ROOT / "tampered_text",
    "copy_move": ROOT / "tampered_copy_move",
    "recompression": ROOT / "tampered_recompression",
    "blur": ROOT / "tampered_blur",
}

print("=== FUNSD GENERATED DATA VERIFICATION ===")

total = 0
leaked = []

for name, directory in directories.items():
    files = sorted(directory.glob("*.png"))
    total += len(files)

    print(f"{name}: {len(files)}")

    for path in files:
        if path.name in final_test_names:
            leaked.append(path.name)

print(f"\nTotal generated files: {total}")
print(f"Final-test leakage: {len(leaked)}")

if leaked:
    print("\nLEAKED FILES:")
    for name in leaked:
        print(name)

print(
    "\nSTATUS:",
    "PASS" if total == 200 and not leaked else "FAIL"
)
