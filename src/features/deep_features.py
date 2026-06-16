import os
import cv2
import pickle
import numpy as np

from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


# =========================================================
# CONFIG
# =========================================================
IMAGE_SIZE = 224
BATCH_SIZE = 32

SUPPORTED_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg"
)

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# LOAD PRETRAINED EFFICIENTNETB0
# =========================================================
def load_model():

    print("\n🚀 Loading EfficientNetB0 Model...")

    model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )

    print("✅ EfficientNetB0 Loaded Successfully!")

    return model


# =========================================================
# LOAD IMAGE
# =========================================================
def load_image(image_path):

    try:

        image = cv2.imread(image_path)

        if image is None:
            return None

        # BGR → RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Resize
        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        # Convert to float32
        image = image.astype(np.float32)

        return image

    except Exception as e:

        print(f"❌ Error Loading Image: {image_path}")
        print(e)

        return None


# =========================================================
# EXTRACT FEATURES FROM BATCH
# =========================================================
def extract_batch_features(
    model,
    image_batch
):

    image_batch = np.array(
        image_batch,
        dtype=np.float32
    )

    # EfficientNet preprocessing
    image_batch = preprocess_input(
        image_batch
    )

    # Feature extraction
    features = model.predict(
        image_batch,
        verbose=0
    )

    return features


# =========================================================
# GET IMAGE FILES
# =========================================================
def get_image_files(folder_path):

    files = [

        file_name

        for file_name in os.listdir(folder_path)

        if file_name.lower().endswith(
            SUPPORTED_EXTENSIONS
        )
    ]

    return files


# =========================================================
# PROCESS DATASET
# =========================================================
def process_dataset(
    model,
    data_dir,
    output_file
):

    classes = {
        "real": 0,
        "fake": 1
    }

    print(f"\n📂 Processing Dataset:")
    print(data_dir)

    image_paths = []
    labels = []

    # =====================================================
    # COLLECT IMAGE PATHS
    # =====================================================
    for class_name, label in classes.items():

        class_path = os.path.join(
            data_dir,
            class_name
        )

        if not os.path.exists(class_path):

            print(f"❌ Missing Folder: {class_path}")
            continue

        files = get_image_files(class_path)

        print(
            f"🔹 {class_name.upper()} Images: "
            f"{len(files)}"
        )

        for file_name in files:

            image_path = os.path.join(
                class_path,
                file_name
            )

            image_paths.append(image_path)

            labels.append(label)

    print(f"\n✅ Total Images Found: {len(image_paths)}")

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================
    total_saved = 0

    with open(output_file, "wb") as f:

        for i in tqdm(
            range(
                0,
                len(image_paths),
                BATCH_SIZE
            ),
            desc="Extracting Features"
        ):

            batch_paths = image_paths[
                i:i + BATCH_SIZE
            ]

            batch_labels = labels[
                i:i + BATCH_SIZE
            ]

            images = []
            valid_labels = []

            # =============================================
            # LOAD IMAGES
            # =============================================
            for image_path, label in zip(
                batch_paths,
                batch_labels
            ):

                image = load_image(image_path)

                if image is not None:

                    images.append(image)

                    valid_labels.append(label)

            # Skip empty batch
            if len(images) == 0:
                continue

            # =============================================
            # EXTRACT FEATURES
            # =============================================
            try:

                features = extract_batch_features(
                    model,
                    images
                )

                # Debug print
                print(
                    f"\n✅ Batch Feature Shape: "
                    f"{features.shape}"
                )

                # =========================================
                # SAVE FEATURES
                # =========================================
                for feat, label in zip(
                    features,
                    valid_labels
                ):

                    pickle.dump(
                        (feat, label),
                        f
                    )

                    total_saved += 1

            except Exception as e:

                print("\n❌ Feature Extraction Error")
                print(e)

    print(f"\n💾 Features Saved:")
    print(output_file)

    print(f"✅ Total Samples Saved: {total_saved}")


# =========================================================
# MAIN
# =========================================================
def main():

    # =====================================================
    # PROJECT ROOT
    # =====================================================
    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    )

    # =====================================================
    # DATASET PATH
    # =====================================================
    DATASET_DIR = os.path.join(
        BASE_DIR,
        "deepfake_dataset",
        "real-vs-fake"
    )

    # =====================================================
    # OUTPUT PATH
    # =====================================================
    OUTPUT_DIR = os.path.join(
        BASE_DIR,
        "src",
        "utils",
        "efficientnet_features"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("\n📂 Dataset Path:")
    print(DATASET_DIR)

    print("\n📂 Output Path:")
    print(OUTPUT_DIR)

    # =====================================================
    # LOAD MODEL
    # =====================================================
    model = load_model()

    # =====================================================
    # TRAIN FEATURES
    # =====================================================
    process_dataset(
        model=model,

        data_dir=os.path.join(
            DATASET_DIR,
            "train"
        ),

        output_file=os.path.join(
            OUTPUT_DIR,
            "train_features.pkl"
        )
    )

    # =====================================================
    # VALID FEATURES
    # =====================================================
    process_dataset(
        model=model,

        data_dir=os.path.join(
            DATASET_DIR,
            "valid"
        ),

        output_file=os.path.join(
            OUTPUT_DIR,
            "valid_features.pkl"
        )
    )

    # =====================================================
    # TEST FEATURES
    # =====================================================
    process_dataset(
        model=model,

        data_dir=os.path.join(
            DATASET_DIR,
            "test"
        ),

        output_file=os.path.join(
            OUTPUT_DIR,
            "test_features.pkl"
        )
    )

    print("\n✅ ALL FEATURES EXTRACTED SUCCESSFULLY!")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    main()