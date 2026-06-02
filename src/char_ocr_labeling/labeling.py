"""
Character labeling module.

Responsibilities:
- Run OCR on segmented character crops
- Normalize valid characters into a standard format
- Compute quality scores for ranking
- Assign structured labels for dataset organization

Pipeline:
1. OCR inference using EasyOCR
2. Filter invalid predictions (non-single characters / low confidence)
3. Normalize character image (size, alignment, noise removal)
4. Compute composite quality score
5. Map label to dataset-friendly folder format

Input:
- segments: List[np.ndarray] → raw character crops

Output:
- List[dict] with:
    {
        "img": normalized 64x64 image,
        "label": formatted label (e.g., small_a, capital_b, 3),
        "conf": OCR confidence,
        "score": computed quality score
    }

Notes:
- Invalid or low-quality crops are skipped
- Designed to feed grouping + selection stages downstream
"""

from .ocr_engine import OCREngine
from .normalization import normalize_char_image
from .scoring import compute_char_score
from .utils import get_label_folder

class Labeler:

    def __init__(self):
        self.ocr = OCREngine()

    def label_segments(self, segments):

        labeled = []

        for img in segments:

            result = self.ocr.get_single_char(img)

            if result is None:
                continue

            label, conf = result

            normalized = normalize_char_image(img, label)

            if normalized is None:
                continue

            score = compute_char_score(normalized, conf)

            labeled.append({
                "img": normalized,
                "label": get_label_folder(label),
                "conf": conf,
                "score": score
            })

        return labeled