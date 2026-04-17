import numpy as np
import cv2
import os
import pickle
from skimage.feature import local_binary_pattern

print("File is running...")

# ==============================
# 1. LBP Feature Extraction
# ==============================


def extract_lbp_features(image):
    """
    Input: image (numpy array)
    Output: feature vector (1D numpy array)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize (same size as FFT teammate — keeps things consistent)
    image = cv2.resize(image, (128, 128))

    # LBP — every pixel gets replaced by its pattern code
    lbp_map = local_binary_pattern(image, P=8, R=1, method='uniform')

    # Histogram — summarize the whole LBP map into fixed-size vector
    n_bins = 10  # for P=8 uniform LBP, there are exactly 10 possible patterns
    hist, _ = np.histogram(
        lbp_map.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True          # normalize so image size doesn't affect the values
    )

    return hist  # 10 numbers


# ==============================
# 2. Process Dataset
# ==============================
def process_dataset(dataset_path):
    features = []
    labels = []

    for label in ["real", "fake"]:
        folder = os.path.join(dataset_path, label)
        print(f"\nChecking folder: {folder}")

        if not os.path.exists(folder):
            print("Folder not found!")
            continue

        files = os.listdir(folder)
        print(f"Total files: {len(files)}")

        count = 0
        for file in files:

            # Only process image files
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Extract features
            feat = extract_lbp_features(img)
            features.append(feat)
            labels.append(0 if label == "real" else 1)
            count += 1

            # Progress update
            if count % 100 == 0:
                print(f"  Processed {count} images in '{label}'")

    return np.array(features), np.array(labels)


# ==============================
# 3. Save Features
# ==============================
def save_features(dataset_path, output_file):
    print(f"\nProcessing dataset: {dataset_path}")
    X, y = process_dataset(dataset_path)
    with open(output_file, "wb") as f:
        pickle.dump((X, y), f)
    print(f"Saved {output_file} with shape {X.shape}")


# ==============================
# 4. Main Execution
# ==============================
if __name__ == "__main__":
    print(" Starting LBP feature extraction pipeline...\n")

    base_path = r"C:\Users\Taniya vaidya\Downloads\archive (12)\real_vs_fake\real-vs-fake"

    # Create output folder
    os.makedirs("features", exist_ok=True)

    # Train
    save_features(
        os.path.join(base_path, "train"),
        "features/train_lbp_features.pkl"
    )

    # Validation
    save_features(
        os.path.join(base_path, "valid"),
        "features/valid_lbp_features.pkl"
    )

    # Test
    save_features(
        os.path.join(base_path, "test"),
        "features/test_lbp_features.pkl"
    )

    print("\n All datasets processed successfully!")
