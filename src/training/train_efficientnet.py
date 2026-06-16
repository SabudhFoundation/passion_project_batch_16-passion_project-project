import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.utils import plot_model

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# =========================================================
# CONFIG
# =========================================================
IMAGE_SIZE = 224
BATCH_SIZE = 32

INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5

UNFREEZE_LAYERS = 20

SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# =========================================================
# PROJECT PATHS
# =========================================================
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "deepfake_dataset",
    "real-vs-fake"
)

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "cnn"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# DATA GENERATORS
# =========================================================
def create_generators():

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=10,
        horizontal_flip=True,
        zoom_range=0.1
    )

    valid_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_data = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    valid_data = valid_datagen.flow_from_directory(
        VALID_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    test_data = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    return train_data, valid_data, test_data


# =========================================================
# BUILD MODEL
# =========================================================
def build_model():

    print("\n🚀 Loading EfficientNetB0...")

    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    print("✅ EfficientNet Loaded")

    # =========================================
    # FREEZE ALL LAYERS
    # =========================================
    for layer in base_model.layers:
        layer.trainable = False

    # =========================================
    # CUSTOM HEAD
    # =========================================
    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(
        256,
        activation='relu'
    )(x)

    x = Dropout(0.5)(x)

    output = Dense(
        1,
        activation='sigmoid'
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=output
    )

    return model, base_model


# =========================================================
# COMPILE MODEL
# =========================================================
def compile_model(model, lr):

    model.compile(
        optimizer=Adam(learning_rate=lr),

        loss='binary_crossentropy',

        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc')
        ]
    )


# =========================================================
# CALLBACKS
# =========================================================
def get_callbacks():

    callbacks = [

        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2
        ),

        ModelCheckpoint(
            filepath=os.path.join(
                MODEL_DIR,
                "efficientnet_best.keras"
            ),
            save_best_only=True
        )
    ]

    return callbacks


# =========================================================
# TRAIN MODEL
# =========================================================
def train_model(
    model,
    train_data,
    valid_data,
    callbacks,
    epochs
):

    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=epochs,
        callbacks=callbacks
    )

    return history


# =========================================================
# UNFREEZE LAST LAYERS
# =========================================================
def unfreeze_layers(base_model):

    print("\n🔥 Fine-Tuning Last Layers")

    for layer in base_model.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True

    print(f"✅ Last {UNFREEZE_LAYERS} Layers Unfrozen")


# =========================================================
# EVALUATE MODEL
# =========================================================
def evaluate_model(model, test_data):

    print("\n📊 Evaluating Model...")

    loss, accuracy, auc = model.evaluate(test_data)

    print(f"\n✅ Test Accuracy : {accuracy:.4f}")
    print(f"✅ Test ROC-AUC  : {auc:.4f}")

    # =========================================
    # PREDICTIONS
    # =========================================
    y_prob = model.predict(test_data)

    y_pred = (y_prob > 0.5).astype(int)

    y_true = test_data.classes

    # =========================================
    # CLASSIFICATION REPORT
    # =========================================
    print("\n📊 Classification Report\n")

    print(
        classification_report(
            y_true,
            y_pred
        )
    )

    # =========================================
    # CONFUSION MATRIX
    # =========================================
    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\n📊 Confusion Matrix\n")

    print(cm)

    # =========================================
    # ROC-AUC
    # =========================================
    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    print(f"\n🔥 Final ROC-AUC Score: {roc_auc:.4f}")

    return accuracy, roc_auc


# =========================================================
# SAVE MODEL
# =========================================================
def save_model(model):

    save_path = os.path.join(
        MODEL_DIR,
        "efficientnet_finetuned.keras"
    )

    model.save(save_path)

    print(f"\n💾 Model Saved: {save_path}")


# =========================================================
# SAVE ARCHITECTURE
# =========================================================
def save_architecture(model):

    plot_model(
        model,
        to_file=os.path.join(
            MODEL_DIR,
            "efficientnet_architecture.png"
        ),
        show_shapes=True,
        show_layer_names=True
    )

    print("\n🧠 Architecture Diagram Saved")


# =========================================================
# PLOT TRAINING CURVES
# =========================================================
def plot_training(history1, history2):

    acc = (
        history1.history['accuracy']
        + history2.history['accuracy']
    )

    val_acc = (
        history1.history['val_accuracy']
        + history2.history['val_accuracy']
    )

    plt.figure(figsize=(10, 5))

    plt.plot(acc, label='Train Accuracy')

    plt.plot(val_acc, label='Validation Accuracy')

    plt.title("EfficientNet Training Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.savefig(
        os.path.join(
            MODEL_DIR,
            "efficientnet_accuracy.png"
        )
    )

    plt.close()

    print("\n📈 Accuracy Graph Saved")


# =========================================================
# MAIN
# =========================================================
def main():

    print("\n🚀 EfficientNet Fine-Tuning Pipeline")

    # =========================================
    # LOAD DATA
    # =========================================
    train_data, valid_data, test_data = create_generators()

    # =========================================
    # BUILD MODEL
    # =========================================
    model, base_model = build_model()

    # =========================================
    # SAVE ARCHITECTURE
    # =========================================
    save_architecture(model)

    # =========================================
    # COMPILE PHASE 1
    # =========================================
    compile_model(model, lr=0.001)

    callbacks = get_callbacks()

    # =========================================
    # PHASE 1
    # =========================================
    print("\n🔥 PHASE 1: Transfer Learning")

    history1 = train_model(
        model,
        train_data,
        valid_data,
        callbacks,
        INITIAL_EPOCHS
    )

    # =========================================
    # UNFREEZE LAYERS
    # =========================================
    unfreeze_layers(base_model)

    # =========================================
    # COMPILE PHASE 2
    # =========================================
    compile_model(model, lr=1e-5)

    # =========================================
    # PHASE 2
    # =========================================
    print("\n🔥 PHASE 2: Fine-Tuning")

    history2 = train_model(
        model,
        train_data,
        valid_data,
        callbacks,
        FINE_TUNE_EPOCHS
    )

    # =========================================
    # EVALUATION
    # =========================================
    accuracy, auc = evaluate_model(
        model,
        test_data
    )

    # =========================================
    # SAVE MODEL
    # =========================================
    save_model(model)

    # =========================================
    # PLOT RESULTS
    # =========================================
    plot_training(
        history1,
        history2
    )

    # =========================================
    # FINAL RESULTS
    # =========================================
    print("\n📊 FINAL RESULTS\n")

    print(f"{'Model':<20} {'Accuracy':<15} {'ROC-AUC':<15}")

    print("-" * 50)

    print(
        f"{'EfficientNetB0':<20}"
        f"{accuracy*100:.2f}%{'':<8}"
        f"{auc:.4f}"
    )

    print("\n✅ EfficientNet Fine-Tuning Completed!")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    main()