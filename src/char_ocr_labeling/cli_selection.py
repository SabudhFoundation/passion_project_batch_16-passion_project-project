"""
CLI-based candidate selection module.

Provides a headless workflow for human-in-the-loop filtering of OCR results.

Responsibilities:
- Group labeled characters by class
- Sort candidates by quality score
- Generate visual grids for inspection
- Allow manual selection of best samples per label

Components:

create_grid:
    - Builds a tiled grayscale image of candidates
    - Overlays index numbers for selection reference

interactive_best_selection_cli:
    - Saves grid images per label to disk
    - Prompts user to choose best candidate via CLI
    - Stores selected samples in data[sub]["best"]

Input:
- data structure:
    {
        sub: {
            "labeled": [ {img, label, score, ...}, ... ]
        }
    }

Output:
- Updated data with:
    data[sub]["best"] = selected samples

Filesystem:
- Temporary grids stored in: src/data/tmp_grids/

Notes:
- Designed for non-GUI environments (e.g., SSH, servers)
- Human selection significantly improves dataset quality
"""

import os
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
TMP_GRID_DIR = DATA_ROOT / "tmp_grids"

def create_grid(candidates, cols=6, cell_size=64):

    total = len(candidates)
    rows = int(np.ceil(total / cols))

    grid = np.ones((rows * cell_size, cols * cell_size), dtype=np.uint8) * 255

    for idx, item in enumerate(candidates):

        r, c = divmod(idx, cols)

        img = cv2.resize(item["img"], (cell_size, cell_size))

        y1, x1 = r * cell_size, c * cell_size

        grid[y1:y1+cell_size, x1:x1+cell_size] = img

        cv2.putText(
            grid,
            str(idx+1),
            (x1+2, y1+12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (128,),
            1
        )

    return grid


def interactive_best_selection_cli(data):

    print("\nHeadless grid selection started")

    TMP_GRID_DIR.mkdir(parents=True, exist_ok=True)

    for sub in data:

        print(f"\nProcessing: {sub}")

        labeled = data[sub]["labeled"]

        groups = {}
        for item in labeled:
            groups.setdefault(item["label"], []).append(item)

        for label in groups:
            groups[label].sort(key=lambda x: x["score"], reverse=True)

        selected = []

        for label in sorted(groups.keys()):

            candidates = groups[label]
            if not candidates:
                continue

            print(f"\nLabel: {label}")

            grid = create_grid(candidates)

            grid_path = TMP_GRID_DIR / f"{sub}_{label}.png"
            cv2.imwrite(str(grid_path), grid)

            print(f"Open this image: {grid_path}")

            total = len(candidates)

            while True:
                choice = input(f"Select (1-{total}) or skip: ").strip().lower()

                if choice == "skip":
                    print(f"Skipped: {label}")
                    break

                if choice.isdigit():
                    idx = int(choice)
                    if 1 <= idx <= total:
                        selected.append(candidates[idx - 1])
                        print(f"Selected: {idx}/{total}")
                        break

                print("Invalid input")

        data[sub]["best"] = selected

    print("\nSelection complete")
    return data