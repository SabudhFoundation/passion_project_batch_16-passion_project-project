import pickle
import json
import time
import os
import numpy as np

from src.models.normalizer import FeatureNormalizer
from src.models.svm_classifier import SVMClassifier
from scipy.stats import skew, kurtosis


# =====================================
# CONFIG
# =====================================
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

FEATURE_DIR = os.path.join(PROJECT_ROOT, "data", "features", "combined")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "svm")

os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================
# LOAD FEATURES
# =====================================
def load_stream(path):
    features, labels = [], []

    with open(path, "rb") as f:
        while True:
            try:
                feat, label = pickle.load(f)
                features.append(feat)
                labels.append(label)
            except EOFError:
                break

    return np.array(features), np.array(labels)


print("📂 Loading features...")

X_train, y_train = load_stream(os.path.join(FEATURE_DIR, "train_features.pkl"))
X_valid, y_valid = load_stream(os.path.join(FEATURE_DIR, "valid_features.pkl"))
X_test,  y_test  = load_stream(os.path.join(FEATURE_DIR, "test_features.pkl"))

print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")


# =====================================
# ADD STAT FEATURES
# =====================================
def add_stat_features(X):
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    skewness = skew(X, axis=1).reshape(-1, 1)
    kurt = kurtosis(X, axis=1).reshape(-1, 1)

    return np.hstack([X, mean, std, skewness, kurt])


print("\n⚙️ Adding statistical features...")

X_train = add_stat_features(X_train)
X_valid = add_stat_features(X_valid)
X_test  = add_stat_features(X_test)

print(f"New feature dim: {X_train.shape[1]}")


# =====================================
# FEATURE CONFIG
# =====================================
LBP_DIM = 112
GLCM_DIM = 40


# =====================================
# NORMALIZATION
# =====================================
print("\n⚙️ Applying normalization...")

norm = FeatureNormalizer(
    lbp_dim=LBP_DIM,
    glcm_dim=GLCM_DIM,
    use_log_fft=True,
    use_energy_norm=True,
    fft_weight=1.0,
    glcm_weight=1.0,
    lbp_scale=True
)

norm.fit(X_train)

X_train = norm.transform(X_train)
X_valid = norm.transform(X_valid)
X_test  = norm.transform(X_test)

norm.save(os.path.join(MODEL_DIR, "normalizer_features.pkl"))

print("✅ Normalization complete")


# =====================================
# HYPERPARAMETER TUNING
# =====================================
print("\n🚀 Hyperparameter tuning...\n")

param_grid = [
    {"gamma": 0.005},
    {"gamma": 0.003},
    {"gamma": 0.002},
    {"gamma": 0.0015},
]

best_acc = 0
best_config = None
best_threshold = 0

for i, params in enumerate(param_grid, 1):

    gamma = params["gamma"]
    print(f"\n🔍 [{i}/{len(param_grid)}] gamma={gamma}")

    svm = SVMClassifier(
        model_type="rbf_approx",
        gamma=gamma,
        rbf_components=4000,

        # 🔥 KEY CHANGE (PCA ENABLED)
        use_pca=True,
        pca_components=120,

        class_weight="balanced",
        random_state=42
    )

    svm.train(X_train, y_train)

    threshold, tuned_acc = svm.tune_threshold(X_valid, y_valid)

    print(f"Threshold tuned accuracy: {tuned_acc:.4f}")

    if tuned_acc > best_acc:
        best_acc = tuned_acc
        best_config = params
        best_threshold = threshold


# =====================================
# BEST PARAMS
# =====================================
print("\n🏆 Best Parameters:")
print(best_config)
print(f"Validation Accuracy = {best_acc:.4f}")
print(f"Best Threshold = {best_threshold:.4f}")


# =====================================
# FINAL TRAINING
# =====================================
print("\n🚀 Training final model...")

X_full = np.vstack((X_train, X_valid))
y_full = np.hstack((y_train, y_valid))

svm = SVMClassifier(
    model_type="rbf_approx",
    gamma=best_config["gamma"],
    rbf_components=4000,
    use_pca=True,
    pca_components=120,
    class_weight="balanced",
    random_state=42
)

svm.train(X_full, y_full)

svm.best_threshold = best_threshold


# =====================================
# FINAL TEST
# =====================================
print("\n🎯 Final Test Performance:")

svm.evaluate(X_test, y_test)


# =====================================
# SAVE MODEL
# =====================================
svm.save(os.path.join(MODEL_DIR, "svm_features.pkl"))

with open(os.path.join(MODEL_DIR, "best_params_features.json"), "w") as f:
    json.dump({
        "gamma": best_config["gamma"],
        "threshold": float(best_threshold)
    }, f, indent=4)

with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
    json.dump({0: "real", 1: "fake"}, f)

print("\n🎉 Pipeline complete!")