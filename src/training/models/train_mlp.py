import pickle
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier


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

    if len(X) == 0:
        raise ValueError(f"No data found in {file_path}")

    return np.array(X, dtype=np.float32), np.array(y)


def train_model(X_train, y_train, X_test, y_test, model=None):

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("Mismatch in training data")

    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError("Mismatch in test data")

    if model is None:
        model = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            solver="adam",
            batch_size=256,
            max_iter=80,
            early_stopping=True,
            verbose=True,
            random_state=42
        )

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    return model, acc


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)


def split_features(X, glcm_dim, lbp_dim):
    total_dim = X.shape[1]

    if glcm_dim + lbp_dim > total_dim:
        raise ValueError("Invalid feature dimensions")

    glcm = X[:, :glcm_dim]
    lbp = X[:, glcm_dim:glcm_dim + lbp_dim]
    fft = X[:, glcm_dim + lbp_dim:]

    return glcm, lbp, fft