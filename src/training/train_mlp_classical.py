import os
import cv2
import numpy as np

from src.feature_engineering.build_features import FeatureBuilder
from src.models.standard_classifier import build_mlp
from src.pipeline.trainer import train_and_save


# ==========================================
# LOAD DATASET
# ==========================================
def load_dataset(dataset_path):
    paths = []
    labels = []

    for label, folder in enumerate(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)

        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            paths.append(os.path.join(folder_path, file))
            labels.append(label)

    return paths, labels


# ==========================================
# MAIN
# ==========================================
def main():

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    dataset_path = os.path.join(
        BASE_DIR,
        "deepfake_dataset",
        "real-vs-fake",
        "train"
    )

    print(f"📂 Using dataset: {dataset_path}")

    # ✅ SAFETY CHECK
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Dataset not found at: {dataset_path}")

    paths, labels = load_dataset(dataset_path)

    builder = FeatureBuilder()

    lbp_features = []
    glcm_features = []
    fft_features = []
    valid_labels = []   # ✅ IMPORTANT FIX

    for path, label in zip(paths, labels):
        img = cv2.imread(path)

        if img is None:
            continue

        features = builder.extract_features(img)

        lbp_features.append(features["lbp"])
        glcm_features.append(features["glcm"])
        fft_features.append(features["fft"])

        valid_labels.append(label)   # ✅ KEEP LABEL IN SYNC

    # Convert to numpy
    lbp = np.array(lbp_features)
    glcm = np.array(glcm_features)
    fft = np.array(fft_features)
    y = np.array(valid_labels)

    print(f"\n📊 Samples Loaded: {len(y)}")

    if len(y) == 0:
        raise ValueError("❌ No valid images found!")

    # ===============================
    # INDIVIDUAL MODELS
    # ===============================
    train_and_save(lbp, y, build_mlp(), "lbp_mlp")
    train_and_save(glcm, y, build_mlp(), "glcm_mlp")
    train_and_save(fft, y, build_mlp(), "fft_mlp")

    # ===============================
    # COMBINED MODEL
    # ===============================
    combined = np.array([
        np.concatenate([l, g, f])
        for l, g, f in zip(lbp, glcm, fft)
    ])

    train_and_save(combined, y, build_mlp(), "combined_mlp")


# ==========================================
if __name__ == "__main__":
    main()