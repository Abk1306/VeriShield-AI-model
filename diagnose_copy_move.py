from pathlib import Path

import cv2


SOURCE = Path("dataset/external/FUNSD_eval/tampered")


def detect_copy_move(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    # ORB is lightweight and works locally without external models.
    orb = cv2.ORB_create(
        nfeatures=3000,
        scaleFactor=1.2,
        nlevels=8,
    )

    keypoints, descriptors = orb.detectAndCompute(image, None)

    if descriptors is None or len(keypoints) < 10:
        return 0, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    matches = matcher.knnMatch(
        descriptors,
        descriptors,
        k=2,
    )

    good_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue

        first, second = pair

        if first.distance < 0.75 * second.distance:
            p1 = keypoints[first.queryIdx].pt
            p2 = keypoints[first.trainIdx].pt

            # Ignore self-matches and very small movements.
            distance = (
                (p1[0] - p2[0]) ** 2
                + (p1[1] - p2[1]) ** 2
            ) ** 0.5

            if distance > 50:
                good_matches.append(first)

    return len(keypoints), len(good_matches)


print("=== COPY-MOVE FEATURE DIAGNOSTIC ===")

files = sorted(SOURCE.glob("*.png"))

for path in files:
    keypoints, matches = detect_copy_move(path)

    print(
        f"{path.name}: "
        f"keypoints={keypoints}, "
        f"duplicate_matches={matches}"
    )
