import os
import pickle
import numpy as np


def train_and_save_model(X_train, y_train, X_valid, y_valid, model, model_name, model_dir):
    
    if len(X_train) == 0:
        print(f"Skipping {model_name} (no data)")
        return None

    print(f"\nTraining {model_name}...")

    # =========================
    # TRAIN
    # =========================
    model.train(X_train, y_train)

    # =========================
    # VALIDATION
    # =========================
    val_acc = model.evaluate(X_valid, y_valid)

    print(f"{model_name} Validation Accuracy: {val_acc:.4f}")

    # =========================
    # SAVE MODEL
    # =========================
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    model.save(model_path)

    print(f"Saved model: {model_path}")

    return {
        "model_name": model_name,
        "val_accuracy": val_acc
    }