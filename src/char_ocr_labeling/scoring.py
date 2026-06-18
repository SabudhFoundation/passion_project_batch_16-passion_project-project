import cv2
import numpy as np
from .config import *

def compute_char_score(img, ocr_conf):
    """
    Compute a quality score for a normalized character image.

    The score combines OCR confidence with geometric and structural
    heuristics to rank candidate character crops.

    Metrics used:
        1. OCR confidence (primary signal)
        2. Fill ratio (foreground pixel density)
        3. Connected components (noise check)
        4. Aspect ratio (shape sanity)
        5. Centering (alignment within canvas)

    Args:
        img (np.ndarray): Normalized binary image (white bg, black fg)
                          Expected shape: (H, W), typically 64x64
        ocr_conf (float): OCR confidence score (0–1)

    Returns:
        float: Weighted score indicating character quality (higher is better)

    Notes:
        - Assumes foreground pixels are < 128
        - Returns 0 if no valid foreground detected
        - Weight constants are defined in config.py
    """
    ink = (img < 128).astype(np.uint8)

    h, w = ink.shape

    # Fill ratio (ideal ≈ 0.18)
    fill_ratio = np.sum(ink) / (h * w)
    fill_score = 1.0 - abs(fill_ratio - 0.18)

    # Connected components (ideal = 1 foreground object)
    nlabels, _, _, _ = cv2.connectedComponentsWithStats(ink, 8)
    cc_score = 1.0 if nlabels <= 3 else 0.5

    # Bounding box + aspect ratio
    coords = cv2.findNonZero(ink)
    if coords is None:
        return 0

    x, y, bw, bh = cv2.boundingRect(coords)
    aspect = bw / (bh + 1e-5)
    aspect_score = 1.0 - abs(aspect - 0.5)

    # Centering
    cx = x + bw // 2
    cy = y + bh // 2

    center_dist = np.sqrt((cx - w//2)**2 + (cy - h//2)**2)
    center_score = 1.0 - (center_dist / (w//2))

    # Final weighted score
    score = (
        W_OCR * ocr_conf +
        W_FILL * fill_score +
        W_CC * cc_score +
        W_ASPECT * aspect_score +
        W_CENTER * center_score
    )

    return score