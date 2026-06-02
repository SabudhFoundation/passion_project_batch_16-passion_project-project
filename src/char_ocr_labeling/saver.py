"""
Dataset saving utilities for OCR pipeline.

This module handles:
    - Saving ONE representative image per label (clean dataset)
    - Saving best-selected characters (flat structure)
    - Generating inverted variants

Output structure:
    <output_dir>/final_dataset/<sub>/<label>/img.jpg
    <output_dir>/best_chars/<sub>/<label>.jpg
    <output_dir>/best_chars_inverted/<sub>/<label>.jpg
"""

import cv2
import numpy as np
from pathlib import Path


# =========================================
# SAFE WRITE
# =========================================
def safe_write(path: Path, img):
    ok = cv2.imwrite(str(path), img)
    if not ok:
        print(f"Failed to save: {path}")


# =========================================
# NORMALIZE LABEL (important for consistency)
# =========================================
def normalize_label(label: str) -> str:
    return label.strip()


# =========================================
# MAIN SAVE FUNCTION
# =========================================
def save_final_outputs(data, output_dir="data"):

    output_dir = Path(output_dir)

    DATASET_ROOT = output_dir / "final_dataset"
    BEST_ROOT = output_dir / "best_chars"
    BEST_INV_ROOT = output_dir / "best_chars_inverted"

    # Create base dirs
    for path in [DATASET_ROOT, BEST_ROOT, BEST_INV_ROOT]:
        path.mkdir(parents=True, exist_ok=True)

    # =========================================
    # PROCESS EACH GROUP (kid1, kid2, etc.)
    # =========================================
    for sub in data:

        print(f"\n💾 Saving group: {sub}")

        dataset_sub = DATASET_ROOT / sub
        best_sub = BEST_ROOT / sub
        best_inv_sub = BEST_INV_ROOT / sub

        dataset_sub.mkdir(parents=True, exist_ok=True)
        best_sub.mkdir(parents=True, exist_ok=True)
        best_inv_sub.mkdir(parents=True, exist_ok=True)

        # =========================================
        # 1. FINAL DATASET (ONE IMAGE PER LABEL)
        # =========================================
        labeled_items = data[sub].get("labeled", [])

        label_seen = set()
        saved_count = 0

        for item in labeled_items:
            label = normalize_label(item["label"])
            img = item["img"]

            # keep only ONE per label
            if label in label_seen:
                continue
            label_seen.add(label)

            label_dir = dataset_sub / label
            label_dir.mkdir(parents=True, exist_ok=True)

            img_path = label_dir / "img.jpg"
            safe_write(img_path, img)

            saved_count += 1

        print(f"   → Saved {saved_count} unique labels")

        # =========================================
        # 2. BEST CHARS (FLAT STRUCTURE)
        # =========================================
        best_items = data[sub].get("best", [])

        if not best_items:
            print(f"   ⚠ No best selected for {sub}")
            continue

        for item in best_items:
            label = normalize_label(item["label"])
            img = item["img"]

            fname = f"{label}.jpg"

            normal_path = best_sub / fname
            inv_path = best_inv_sub / fname

            # Avoid overwrite
            if normal_path.exists():
                fname = f"{label}_{np.random.randint(1e6)}.jpg"
                normal_path = best_sub / fname
                inv_path = best_inv_sub / fname

            # Save normal
            safe_write(normal_path, img)

            # Save inverted
            inv_img = cv2.bitwise_not(img)
            safe_write(inv_path, inv_img)

            print(f"   ✔ Saved best: {sub}/{fname}")

    print("\nDataset saving complete")