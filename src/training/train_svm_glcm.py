import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV

import joblib
from tqdm import tqdm

print("Starting Optimized SVM Pipeline...")

# =====================================
# DATASET PATH
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.abspath(
    os.path.join(BASE_DIR, "../../deepfake_dataset/real-vs-fake")
)

print("Dataset Path:", base_path)


# =====================================
# GLCM FEATURE EXTRACTION
# =====================================
def extract_glcm_features(image):

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (128, 128))

    glcm = graycomatrix(
        image,
        distances=[1, 2],
        angles=[0, np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    features = []

    for prop in ['contrast', 'energy', 'homogeneity', 'correlation']:
        features.extend(graycoprops(glcm, prop).flatten())

    return np.array(features)


# =====================================
# LOAD DATASET (LIMITED FOR SPEED)
# =====================================
def load_dataset(path, max_images=15000):

    features = []
    labels = []

    if not os.path.exists(path):
        print(f"Dataset path not found: {path}")
        return np.array(features), np.array(labels)

    for label in ["real", "fake"]:
        folder = os.path.join(path, label)

        print(f"\nReading {folder}")

        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        # 🔥 LIMIT DATA (IMPORTANT)
        files = files[:max_images]

        for file in tqdm(files, desc=f"{label} processing"):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            feat = extract_glcm_features(img)
            features.append(feat)
            labels.append(0 if label == "real" else 1)

    return np.array(features), np.array(labels)


# =====================================
# LOAD DATA
# =====================================
X_train, y_train = load_dataset(os.path.join(base_path, "train"), max_images=15000)
X_test, y_test = load_dataset(os.path.join(base_path, "test"), max_images=5000)

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

if len(X_train) == 0 or len(X_test) == 0:
    print("Dataset loading failed.")
    exit()


# =====================================
# NORMALIZATION
# =====================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data Normalized")


# =====================================
# TRAIN-VALID SPLIT
# =====================================
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)


# =====================================
# RANDOMIZED SEARCH (FAST + GOOD)
# =====================================
print("Running RandomizedSearch (optimized)...")

param_dist = {
    'C': [1, 10, 50],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']
}

search = RandomizedSearchCV(
    SVC(probability=True),
    param_distributions=param_dist,
    n_iter=4,          # 🔥 only 4 combinations
    cv=2,              # 🔥 reduce folds
    verbose=2,
    n_jobs=2           # 🔥 limit CPU usage (important)
)

search.fit(X_tr, y_tr)

model = search.best_estimator_

print("Best Parameters:", search.best_params_)


# =====================================
# VALIDATION
# =====================================
y_val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)

print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")


# =====================================
# TEST
# =====================================
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# =====================================
# SAVE MODEL
# =====================================
model_dir = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "svm_glcm_model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

print(f"Model saved in: {model_dir}")