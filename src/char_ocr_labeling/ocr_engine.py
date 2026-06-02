"""
OCR engine wrapper using EasyOCR.

Responsibilities:
- Preprocess character crops for OCR (color normalization, upscaling, denoising)
- Run EasyOCR inference
- Filter predictions to retain only high-confidence single-character outputs

Key behaviors:
- Upscales input image to improve OCR accuracy on small glyphs
- Applies light Gaussian blur to stabilize predictions
- Filters non-alphanumeric and multi-character outputs
- Returns the best candidate based on confidence threshold

Returns:
- (char, confidence) tuple OR None if no valid prediction
"""

import cv2
import easyocr
from .config import OCR_CONF

class OCREngine:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)

    def get_single_char(self, img):

        # -----------------------------
        # PREPROCESS FOR OCR
        # -----------------------------
        if len(img.shape) == 2:
            proc = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            proc = img.copy()

        # 🔥 upscale (critical)
        proc = cv2.resize(proc, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

        # 🔥 slight blur helps OCR stability
        proc = cv2.GaussianBlur(proc, (3, 3), 0)

        results = self.reader.readtext(proc)

        print("OCR RAW:", results)

        best = None

        for _, text, conf in results:

            text = "".join(filter(str.isalnum, text))

            if len(text) != 1:
                continue

            if conf < OCR_CONF:
                continue

            if best is None or conf > best[1]:
                best = (text, conf)

        return best