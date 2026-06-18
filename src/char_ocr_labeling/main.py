"""
OCR pipeline orchestrator.

Responsibilities:
- Execute end-to-end character labeling on segmented crops
- Aggregate labeled outputs into structured groups for downstream use

Pipeline:
1. Initialize Labeler (OCR + normalization + scoring)
2. Label each character crop
3. Group labeled samples by class label
4. Sort each group by descending quality score

Input:
- char_crops: List[np.ndarray]
    Raw segmented character images

Output:
- dict:
    {
        "labeled": List[dict],   # flat labeled results
        "groups": {
            label: List[dict]    # grouped + sorted by score
        }
    }

Notes:
- Grouping enables UI-based selection of best samples
- Sorting ensures highest-quality candidates appear first
- Designed to integrate with Streamlit frontend and dataset saver
"""

from .labeling import Labeler

def run_ocr_pipeline(char_crops):
    """
    Accepts list of character images (np.ndarray)
    Returns labeled data grouped for UI selection
    """

    labeler = Labeler()

    labeled = labeler.label_segments(char_crops)

    # group by label
    groups = {}
    for item in labeled:
        groups.setdefault(item["label"], []).append(item)

    # sort by score
    for label in groups:
        groups[label].sort(key=lambda x: x["score"], reverse=True)

    return {
        "labeled": labeled,
        "groups": groups
    }