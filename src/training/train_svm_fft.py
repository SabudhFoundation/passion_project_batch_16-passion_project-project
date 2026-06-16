import pickle
import json
import time
import os
import numpy as np

from src.models.normalizer import FeatureNormalizer
from src.models.svm_classifier import SVMClassifier


# =====================================
# PATH SETUP
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

FEATURE_DIR = os.path.join(PROJECT_ROOT, "data", "features", "fft_only")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "svm")

os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================
# 1. LOAD FEATURES
# =====================================
print("📂 Loading FFT features...")

# Each file contains: (features, labels)
with open(os.path.join(FEATURE_DIR, "train_fft_features.pkl"), "rb") as f:
    X_train, y_train = pickle.load(f)

with open(os.path.join(FEATURE_DIR, "valid_fft_features.pkl"), "rb") as f:
    X_valid, y_valid = pickle.load(f)

with open(os.path.join(FEATURE_DIR, "test_fft_features.pkl"), "rb") as f:
    X_test, y_test = pickle.load(f)

print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")


# =====================================
# 2. NORMALIZATION (FFT ONLY)
# =====================================
print("\n⚙️ Applying FFT preprocessing...")

# 🔥 Since only FFT features → treat whole vector as FFT
fft_dim = X_train.shape[1]

norm = FeatureNormalizer(
    lbp_dim=0,               # no LBP
    glcm_dim=0,              # no GLCM
    use_log_fft=True,        # reduce skew
    use_energy_norm=True,    # normalize per sample
    fft_weight=1.0
)

# Fit only on training data (IMPORTANT)
norm.fit(X_train)

# Transform all datasets
X_train = norm.transform(X_train)
X_valid = norm.transform(X_valid)
X_test  = norm.transform(X_test)

# Save normalizer (used during inference)
norm.save(os.path.join(FEATURE_DIR, "normalizer.pkl"))

print("✅ FFT preprocessing complete")


# =====================================
# 3. HYPERPARAMETER TUNING
# =====================================
print("\n🚀 Starting RBF Approx tuning...\n")

gamma_values = [0.02, 0.01, 0.005]

best_acc = 0
best_gamma = None

for i, gamma in enumerate(gamma_values, 1):

    print(f"\n🔍 [{i}/{len(gamma_values)}] Testing gamma={gamma}")

    start_time = time.time()

    # 🔥 Build model using modular pipeline
    svm = SVMClassifier(
        model_type="rbf_approx",
        gamma=gamma,
        rbf_components=4000,
        use_pca=True,
        pca_components=100,
        class_weight="balanced"
    )

    # Train model
    svm.train(X_train, y_train)

    # Validate
    acc = svm.evaluate(X_valid, y_valid)

    print(f"⏱ Time: {time.time() - start_time:.2f} sec")

    if acc > best_acc:
        best_acc = acc
        best_gamma = gamma


# =====================================
# 4. BEST PARAMETERS
# =====================================
print("\n🏆 Best Parameter Found:")
print(f"gamma = {best_gamma}")
print(f"Validation Accuracy = {best_acc:.4f}")


# =====================================
# 5. FINAL TRAINING (TRAIN + VALID)
# =====================================
print("\n🚀 Training final model on full dataset...")

X_full = np.vstack((X_train, X_valid))
y_full = np.hstack((y_train, y_valid))

start_time = time.time()

svm = SVMClassifier(
    model_type="rbf_approx",
    gamma=best_gamma,
    rbf_components=4000,
    use_pca=True,
    pca_components=100,
    class_weight="balanced"
)

svm.train(X_full, y_full)

print(f"⏱ Final training time: {time.time() - start_time:.2f} sec")


# =====================================
# 6. FINAL TEST
# =====================================
print("\n🎯 Final Test Performance:")

svm.evaluate(X_test, y_test)


# =====================================
# 7. SAVE MODEL + METADATA
# =====================================

# Save trained model
svm.save(os.path.join(MODEL_DIR, "svm_fft.pkl"))

# Save best hyperparameters
with open(os.path.join(MODEL_DIR, "best_params.json"), "w") as f:
    json.dump({"gamma": best_gamma}, f)

# Save label mapping (useful for inference)
with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
    json.dump({0: "real", 1: "fake"}, f)

print("\n🎉 Pipeline complete!")