"""
Character normalization module.

Responsibilities:
- Convert raw character crops into standardized 64x64 glyph images
- Remove noise and retain only the primary connected component
- Normalize stroke thickness adaptively
- Align characters using baseline detection
- Center and scale glyphs with consistent fill ratio

Key steps:
1. Grayscale + Otsu binarization
2. Foreground isolation (largest component)
3. Stroke refinement (adaptive dilation)
4. Baseline detection for vertical alignment
5. Resize with controlled fill ratio
6. Placement on fixed-size canvas

Returns:
- Clean, aligned 64x64 binary character image
- None if glyph extraction fails
"""

import cv2
import numpy as np
from .config import IMG_SIZE, FILL_RATIO


def normalize_char_image(img, label=None, size=IMG_SIZE):

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(th) < 127:
        th = cv2.bitwise_not(th)

    fg = (th == 0).astype(np.uint8)

    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)

    if nlabels <= 1:
        return None

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    clean_fg = np.zeros_like(fg)
    clean_fg[labels == largest_idx] = 1

    char_mask = (clean_fg * 255).astype(np.uint8)
    th = 255 - char_mask

    coords = cv2.findNonZero(char_mask)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    char = th[y:y+h, x:x+w]

    # Stroke thickening
    ink = (char < 128).astype(np.uint8)
    area = np.sum(ink)

    if area < 150:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        iterations = 2
    elif area < 400:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        iterations = 1
    else:
        kernel = None
        iterations = 0

    if kernel is not None:
        ink = cv2.dilate(ink, kernel, iterations=iterations)

    char = 255 - (ink * 255)
    char = cv2.medianBlur(char, 3)

    # Baseline
    ink = (char < 128).astype(np.uint8)
    row_sum = ink.sum(axis=1)

    if np.max(row_sum) == 0:
        baseline = h - 1
    else:
        baseline = np.argmax(row_sum[int(h*0.5):]) + int(h*0.5)

    # Resize
    scale = min((size * FILL_RATIO) / w, (size * FILL_RATIO) / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    char_resized = cv2.resize(char, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    baseline_scaled = int(baseline * scale)

    canvas = np.ones((size, size), dtype=np.uint8) * 255

    center_x = size // 2
    target_baseline = int(size * 0.78)

    x1 = center_x - new_w // 2
    y1 = target_baseline - baseline_scaled

    x1 = max(0, min(size - new_w, x1))
    y1 = max(0, min(size - new_h, y1))

    canvas[y1:y1+new_h, x1:x1+new_w] = char_resized

    return canvas