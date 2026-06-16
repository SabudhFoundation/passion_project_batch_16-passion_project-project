import os
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)
from sklearn.neural_network import MLPClassifier


# ==========================================
# LOAD FEATURES
# ==========================================
def load_features(file_path):
    X, y = [], []

    with open(file_path, "rb") as f:
        try:
            while True:
                feat, label = pickle.load(f)
                X.append(feat)
                y.append(label)
        except EOFError:
            pass

    return np.array(X, dtype=np.float32), np.array(y)


# ==========================================
# TRAIN + VALIDATE + TEST
# ==========================================
def train_model(X_train, y_train, X_val, y_val, X_test, y_test, name):

    print(f"\n🚀 Training {name}")

    # ==========================================
    # SCALING
    # ==========================================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ==========================================
    # MODEL
    # ==========================================
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        batch_size=256,
        max_iter=80,
        early_stopping=True,
        verbose=False,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ==========================================
    # VALIDATION
    # ==========================================
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]

    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_prob)

    # ==========================================
    # TEST
    # ==========================================
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_prob)

    # ==========================================
    # PRINT METRICS
    # ==========================================
    print(f"\n📊 {name}")

    print("\n--- Validation ---")
    print(f"Accuracy: {val_acc:.4f}")
    print(f"ROC-AUC: {val_auc:.4f}")
    print(classification_report(y_val, y_val_pred))

    print("\n--- Test ---")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"ROC-AUC: {test_auc:.4f}")
    print(classification_report(y_test, y_test_pred))

    # ==========================================
    # SAVE MODEL
    # ==========================================
    save_dir = os.path.join("models", "mlp")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, f"{name.lower()}.pkl"), "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    return val_acc, test_acc, val_auc, test_auc


# ==========================================
# MAIN
# ==========================================
def main():

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    FEATURE_DIR = os.path.join(
        BASE_DIR,
        "src",
        "utils",
        "combined_features"
    )

    print("📂 Loading features...")

    X_train, y_train = load_features(os.path.join(FEATURE_DIR, "train_features.pkl"))
    X_valid, y_valid = load_features(os.path.join(FEATURE_DIR, "valid_features.pkl"))
    X_test, y_test = load_features(os.path.join(FEATURE_DIR, "test_features.pkl"))

    # ==========================================
    # FEATURE SPLITTING
    # ==========================================
    glcm_dim = 24
    lbp_dim = 128

    # TRAIN
    glcm_train = X_train[:, :glcm_dim]
    lbp_train = X_train[:, glcm_dim:glcm_dim + lbp_dim]
    fft_train = X_train[:, glcm_dim + lbp_dim:]

    # VALID
    glcm_val = X_valid[:, :glcm_dim]
    lbp_val = X_valid[:, glcm_dim:glcm_dim + lbp_dim]
    fft_val = X_valid[:, glcm_dim + lbp_dim:]

    # TEST
    glcm_test = X_test[:, :glcm_dim]
    lbp_test = X_test[:, glcm_dim:glcm_dim + lbp_dim]
    fft_test = X_test[:, glcm_dim + lbp_dim:]

    # ==========================================
    # TRAIN MODELS
    # ==========================================
    results = []

    results.append(("GLCM",) + train_model(glcm_train, y_train, glcm_val, y_valid, glcm_test, y_test, "glcm"))
    results.append(("LBP",) + train_model(lbp_train, y_train, lbp_val, y_valid, lbp_test, y_test, "lbp"))
    results.append(("FFT",) + train_model(fft_train, y_train, fft_val, y_valid, fft_test, y_test, "fft"))
    results.append(("Combined",) + train_model(X_train, y_train, X_valid, y_valid, X_test, y_test, "combined"))

    # ==========================================
    # FINAL TABLE
    # ==========================================
    print("\n\n📊 FINAL RESULTS TABLE\n")
    print(f"{'Features':<12} {'Val Acc':<10} {'Test Acc':<10} {'Val AUC':<10} {'Test AUC':<10}")
    print("-" * 60)

    for feat, val_acc, test_acc, val_auc, test_auc in results:
        print(f"{feat:<12} {val_acc*100:.2f}%    {test_acc*100:.2f}%    {val_auc:.4f}    {test_auc:.4f}")


# ==========================================
if __name__ == "__main__":
    main()