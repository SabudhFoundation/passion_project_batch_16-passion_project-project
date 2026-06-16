import cv2
import numpy as np
import os
import pickle
from tqdm import tqdm


# ===============================
# FFT FEATURE FUNCTION (your code)
# ===============================
def extract_fft_features(
    image,
    img_size=128,
    bands=None,
    hf_threshold=30
):
    if image is None:
        return None

    if bands is None:
        bands = [
            (0, 8),
            (8, 16),
            (16, 30),
            (30, 45),
            (45, 64),
        ]

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (img_size, img_size)).astype(np.float32) / 255.0

    h, w = image.shape
    ch, cw = h // 2, w // 2

    hann = np.outer(np.hanning(h), np.hanning(w))
    image = image * hann

    fshift = np.fft.fftshift(np.fft.fft2(image))

    magnitude = np.log1p(np.abs(fshift))
    phase = np.angle(fshift)

    features = []

    y, x = np.ogrid[:h, :w]
    r = np.hypot(x - cw, y - ch)

    for r1, r2 in bands:
        mask = (r >= r1) & (r < r2)
        region = magnitude[mask]

        if region.size == 0:
            features.extend([0, 0, 0])
        else:
            features.extend([
                np.mean(region),
                np.std(region),
                np.sum(region ** 2),
            ])

    hf = magnitude[r >= hf_threshold]
    lf = magnitude[r < hf_threshold]

    features.append(np.sum(hf ** 2) / (np.sum(lf ** 2) + 1e-8))

    features.extend([
        magnitude.mean(),
        magnitude.std(),
    ])

    features.extend([
        np.std(phase),
        np.mean(np.abs(phase)),
    ])

    return np.array(features, dtype=np.float32)


# ===============================
# DATASET PROCESSING + SAVING
# ===============================
def extract_and_save(dataset_path, output_file):
    X, y = [], []

    for label_name, label_val in [("real", 0), ("fake", 1)]:
        folder = os.path.join(dataset_path, label_name)

        print(f"\nProcessing: {folder}")

        if not os.path.isdir(folder):
            print("❌ Folder missing")
            continue

        files = os.listdir(folder)

        for file in tqdm(files, desc=label_name):
            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            path = os.path.join(folder, file)
            img = cv2.imread(path)

            if img is None:
                continue

            feat = extract_fft_features(img)

            if feat is not None:
                X.append(feat)
                y.append(label_val)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    if len(X) == 0:
        print("⚠️ No features extracted!")
        return

    # SAVE
    with open(output_file, "wb") as f:
        pickle.dump((X, y), f)

    print(f"\n✅ Saved features to: {output_file}")
    print(f"Shape: {X.shape}")


# ===============================
# USAGE (CALL THIS FROM ANYWHERE)
# ===============================
# Example:
# extract_and_save("dataset/train", "train_features.pkl")