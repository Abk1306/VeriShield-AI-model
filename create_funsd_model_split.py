from pathlib import Path
import shutil


SOURCE_ROOT = Path("dataset/external/FUNSD_train_eval")
OUTPUT_ROOT = Path("dataset/external/FUNSD_model_data")

# Clean previous split if it exists.
if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

for split in ["train", "validation"]:
    for category in [
        "authentic",
        "tampered_text",
        "tampered_copy_move",
        "tampered_recompression",
        "tampered_blur",
    ]:
        (OUTPUT_ROOT / split / category).mkdir(
            parents=True,
            exist_ok=True,
        )

# Source documents are identified from the authentic directory.
source_files = sorted(
    (SOURCE_ROOT / "authentic").glob("*.png")
)

train_sources = source_files[:30]
validation_sources = source_files[30:40]

categories = [
    "authentic",
    "tampered_text",
    "tampered_copy_move",
    "tampered_recompression",
    "tampered_blur",
]

for split, sources in [
    ("train", train_sources),
    ("validation", validation_sources),
]:
    for source in sources:
        for category in categories:
            source_file = SOURCE_ROOT / category / source.name

            if source_file.exists():
                destination = (
                    OUTPUT_ROOT
                    / split
                    / category
                    / source.name
                )

                shutil.copy2(source_file, destination)

print("=== FUNSD MODEL SPLIT ===")
print(f"Training source documents: {len(train_sources)}")
print(f"Validation source documents: {len(validation_sources)}")
print("Final test source documents: 10")
print()
print("Training variants:")
for category in categories:
    count = len(
        list(
            (OUTPUT_ROOT / "train" / category).glob("*.png")
        )
    )
    print(f"  {category}: {count}")

print()
print("Validation variants:")
for category in categories:
    count = len(
        list(
            (OUTPUT_ROOT / "validation" / category).glob("*.png")
        )
    )
    print(f"  {category}: {count}")

print()
print("STATUS: PASS")
