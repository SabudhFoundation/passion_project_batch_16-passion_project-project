import numpy as np
import cv2
import os
import pickle
from skimage.feature import graycomatrix, graycoprops

print("GLCM file is running...")

# ================================
# 1. GLCM Feature Extraction
# ================================

def extract_glcm_features(image):
    """
    Input: image (numpy array)
    Output: feature vector (1D numpy array)
    """

    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize (same as LBP for consistency)
    image = cv2.resize(image, (128, 128))

    # Compute GLCM
    glcm = graycomatrix(
        image,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    # Extract features
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    return np.array([contrast, energy, homogeneity, correlation])


# ================================
# 2. Process Dataset
# ================================

def process_dataset(dataset_path):
    features = []
    labels = []

    for label in ["real", "fake"]:
        folder = os.path.join(dataset_path, label)
        print(f"Checking folder: {folder}")

        if not os.path.exists(folder):
            print("Folder not found")
            continue

        files = os.listdir(folder)
        print(f"Total files: {len(files)}")

        count = 0

        for file in files:
            # Only process image files
            if not file.endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Extract features
            feat = extract_glcm_features(img)
            features.append(feat)
            labels.append(0 if label == "real" else 1)

            count += 1

            # Progress update
            if count % 100 == 0:
                print(f"Processed {count} images in {label}")

    return np.array(features), np.array(labels)


# ================================
# 3. Save Features
# ================================

def save_features(dataset_path, output_file):
    print(f"\nProcessing dataset: {dataset_path}")

    X, y = process_dataset(dataset_path)

    with open(output_file, "wb") as f:
        pickle.dump((X, y), f)

    print(f"Saved {output_file} with shape {X.shape}")


# ================================
# 4. Main Execution
# ================================

if __name__ == "__main__":
    print("Starting GLCM feature extraction pipeline...\n")

    base_path = "real-vs-fake"

    # Create output folder
    os.makedirs("features", exist_ok=True)

    # Train
    save_features(
        os.path.join(base_path, "train"),
        "features/train_glcm_features.pkl"
    )

    # Validation
    save_features(
        os.path.join(base_path, "valid"),
        "features/valid_glcm_features.pkl"
    )

    # Test
    save_features(
        os.path.join(base_path, "test"),
        "features/test_glcm_features.pkl"
    )

    print("\nAll datasets processed successfully!")