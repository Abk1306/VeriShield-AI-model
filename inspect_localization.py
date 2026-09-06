from pathlib import Path

import cv2

from verishield_ai.preprocessing import preprocess_document
from verishield_ai.forensic.ela import compute_ela
from verishield_ai.forensic.noise import compute_noise_map
from verishield_ai.forensic.compression import compute_compression_map
from verishield_ai.localization import create_anomaly_map, save_anomaly_map


tests = {
    "authentic": "dataset/authentic/test_document.png",
    "tampered": "dataset/generated/tampered/copy_move_000.jpg",
}

output_dir = Path("outputs/anomaly_maps")
output_dir.mkdir(parents=True, exist_ok=True)

for name, path in tests.items():
    image = preprocess_document(path)

    ela = compute_ela(image)
    noise = compute_noise_map(image)
    compression = compute_compression_map(image)

    anomaly = create_anomaly_map(
        ela,
        noise,
        compression,
    )

    output_path = output_dir / f"{name}_anomaly.png"
    save_anomaly_map(anomaly, str(output_path))

    print(f"{name}: {output_path}")
    print(f"shape={anomaly.shape}, min={anomaly.min()}, max={anomaly.max()}, mean={anomaly.mean():.2f}")
