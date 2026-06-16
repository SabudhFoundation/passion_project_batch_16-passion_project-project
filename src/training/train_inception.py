import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

from src.training.train_resnet import get_data_generators


# ==========================================
# MODEL
# ==========================================
def build_inception():

    base_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze base
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model, base_model


# ==========================================
# EVALUATION FUNCTION
# ==========================================
def evaluate(generator, model, name="Validation"):

    print(f"\n📊 {name} Results")

    preds = model.predict(generator)
    preds_binary = (preds > 0.5).astype(int)

    y_true = generator.classes

    print("\nClassification Report\n")
    print(classification_report(y_true, preds_binary))

    auc = roc_auc_score(y_true, preds)
    print(f"ROC-AUC: {auc:.4f}")

    if name == "Test":
        print("\nConfusion Matrix\n")
        print(confusion_matrix(y_true, preds_binary))

    return auc


# ==========================================
# TRAIN PIPELINE
# ==========================================
def train():

    BASE_DIR = "deepfake_dataset/real-vs-fake"

    print("\n🚀 InceptionV3 Training Pipeline")

    train_gen, val_gen, test_gen = get_data_generators(BASE_DIR)

    model, base_model = build_inception()

    # =========================
    # PHASE 1
    # =========================
    print("\n🔥 Phase 1: Transfer Learning")

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=3
    )

    # =========================
    # PHASE 2 (FINE-TUNING)
    # =========================
    print("\n🔥 Phase 2: Fine-Tuning")

    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=3
    )

    # =========================
    # VALIDATION EVALUATION
    # =========================
    val_auc = evaluate(val_gen, model, "Validation")

    # =========================
    # TEST EVALUATION
    # =========================
    test_auc = evaluate(test_gen, model, "Test")

    # =========================
    # FINAL SUMMARY
    # =========================
    print("\n📊 FINAL SUMMARY")
    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    # =========================
    # SAVE MODEL
    # =========================
    save_path = "models/cnn/inceptionv3_finetuned.keras"
    model.save(save_path)

    print(f"\n💾 Model Saved: {save_path}")


# ==========================================
if __name__ == "__main__":
    train()