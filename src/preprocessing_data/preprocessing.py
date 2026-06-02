import cv2
import numpy as np

# ========= CONFIG =========
MIN_NOISE_AREA = 35


# =========================================
# IMAGE UTILITIES
# =========================================
def invert_image(img):
    return cv2.bitwise_not(img)


def remove_noise_floor(img, min_area=MIN_NOISE_AREA):
    """
    Removes small connected components (noise)
    """
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(img, 8)
    cleaned = np.zeros_like(img)

    for i in range(1, nlabels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned


# =========================================
# LINE REMOVAL (SAFE INPAINT)
# =========================================
def inpaint_lines_safe(gray):
    """
    Removes horizontal and vertical lines using morphology + inpainting
    """

    # Step 1: Background normalization
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (101, 101))
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, se)
    diff = cv2.divide(gray, bg, scale=255)

    # Step 2: Threshold for line detection
    line_thresh = cv2.adaptiveThreshold(
        diff, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10
    )

    # Step 3: Horizontal lines
    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    horiz_lines = cv2.morphologyEx(line_thresh, cv2.MORPH_OPEN, horiz_k)

    # Step 4: Vertical candidates
    vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    vert_candidates = cv2.morphologyEx(line_thresh, cv2.MORPH_OPEN, vert_k)

    # Step 5: Filter vertical lines
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vert_candidates, 8)
    vert_lines = np.zeros_like(vert_candidates)

    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        aspect_ratio = h / (w + 1e-5)

        if (
            h > 60 and
            w < 10 and
            aspect_ratio > 5 and
            area > 200
        ):
            vert_lines[labels == i] = 255

    # Step 6: Combine masks
    lines = cv2.bitwise_or(horiz_lines, vert_lines)
    mask = cv2.dilate(lines, np.ones((3, 3), np.uint8))

    # Step 7: Inpaint
    healed = cv2.inpaint(diff, mask, 1, cv2.INPAINT_TELEA)

    return healed


# =========================================
# MAIN PREPROCESS FUNCTION
# =========================================
def preprocess_single(img, return_bgr=True):
    """
    Full preprocessing pipeline:
    - grayscale
    - line removal
    - binarization
    - noise removal

    Args:
        return_bgr (bool): if True → returns 3-channel image (for segmentation)
                           if False → returns binary (for debugging)
    """

    if img is None:
        raise ValueError("Input image is None")

    # Step 1: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Remove lines
    clean = inpaint_lines_safe(gray)

    # Step 3: Blur
    blurred = cv2.GaussianBlur(clean, (3, 3), 0)

    # Step 4: Adaptive threshold (INVERTED)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 15
    )

    # Step 5: Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    final = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Step 6: Remove small noise
    final = remove_noise_floor(final)

    # =========================================
    # CRITICAL FIX FOR YOUR PIPELINE
    # =========================================
    # segmentation.py expects a "natural image"
    # If you pass pure binary → it reprocesses badly
    #
    # So convert back to 3-channel BGR
    # =========================================
    if return_bgr:
        final_bgr = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
        return final_bgr

    return final