"""
===========================================================
SVG TO TTF FONT GENERATION 
===========================================================

Overview:
---------
This module provides a reusable pipeline to generate a
TrueType Font (TTF) from a dataset of SVG glyph files.

The pipeline:
    1. Loads SVG glyphs from multiple folders
    2. Maps filenames to Unicode characters
    3. Selects the best glyph using a scoring heuristic
    4. Normalizes glyph size and alignment
    5. Applies spacing rules
    6. Fills missing characters with placeholders
    7. Exports the final TTF font

-----------------------------------------------------------
Main Function:
-----------------------------------------------------------
build_font(base_folder, folders, output_ttf, font_name, delete_svgs=False)

-----------------------------------------------------------
Supported Filename Formats:
-----------------------------------------------------------
- capital_a.svg  → 'A'
- small_b.svg    → 'b'
- a.svg          → 'a'
- 5.svg          → '5'

-----------------------------------------------------------
Output:
-----------------------------------------------------------
- Generated .ttf font file
- Optional deletion of SVG input folders

===========================================================
"""

import fontforge
import psMat
from pathlib import Path
import shutil


# ── TYPOGRAPHY CONSTANTS ─────────────────────────────────

UPM = 1000
ASCENDER = 800
DESCENDER = -200

X_HEIGHT = 500
CAP_HEIGHT = 700
ASCENDER_HEIGHT = 750
DESCENDER_DEPTH = -200

ASCENDER_CHARS = set("bdfhijklt")
DESCENDER_CHARS = set("gjpqy")
X_HEIGHT_CHARS = set("aceimnorsuvwxz")


# ── METRICS ─────────────────────────────────────────────

def get_target_metrics(cp):
    """
    Determine vertical alignment targets for a glyph.

    Args:
        cp (int): Unicode codepoint of the character

    Returns:
        tuple:
            (top, bottom) alignment values used for scaling and positioning
    """
    char = chr(cp)

    if char.isupper():
        return CAP_HEIGHT, 0
    if char in ASCENDER_CHARS:
        return ASCENDER_HEIGHT, 0
    if char in DESCENDER_CHARS:
        return X_HEIGHT, DESCENDER_DEPTH
    if char in X_HEIGHT_CHARS:
        return X_HEIGHT, 0
    if char.isdigit():
        return CAP_HEIGHT, 0

    return X_HEIGHT, 0


# ── FILENAME PARSER ─────────────────────────────────────

def filename_to_codepoint(filename):
    """
    Convert a glyph filename into its Unicode codepoint.

    Supported formats:
        capital_a.svg → 'A'
        small_b.svg   → 'b'
        a.svg         → 'a'
        5.svg         → '5'

    Args:
        filename (str): SVG filename

    Returns:
        int or None:
            Unicode codepoint if valid, otherwise None
    """
    stem = Path(filename).stem.lower()
    parts = stem.split("_")

    if parts[0] == "capital" and len(parts) > 1:
        return ord(parts[1].upper())

    if parts[0] == "small" and len(parts) > 1:
        return ord(parts[1].lower())

    if len(stem) == 1 and stem.isalpha():
        return ord(stem)

    if stem.isdigit():
        return ord(stem)

    return None


# ── GLYPH SCORING ───────────────────────────────────────

def score_glyph(glyph):
    """
    Evaluate glyph quality to select the best candidate.

    The scoring considers:
        - Aspect ratio consistency
        - Shape compactness
        - Contour complexity
        - Width-height balance

    Args:
        glyph (fontforge.glyph): Temporary glyph object

    Returns:
        float: Score value (higher is better)
    """
    bb = glyph.boundingBox()
    if bb == (0, 0, 0, 0):
        return -999

    xmin, ymin, xmax, ymax = bb
    w = xmax - xmin
    h = ymax - ymin

    if w == 0 or h == 0:
        return -999

    aspect_ratio = w / h

    return (
        -abs(aspect_ratio - 0.6)
        + min(w, h) / 100
        - abs(len(glyph.foreground) - 2)
        - abs(w - h) / 200
    )


# ── CLEANING ────────────────────────────────────────────

def clean_glyph(glyph):
    """
    Apply safe geometric cleanup operations.

    Operations include:
        - Simplification
        - Extrema addition
        - Contour normalization
        - Rounding

    Args:
        glyph (fontforge.glyph): Glyph to clean
    """
    try:
        glyph.simplify(1.0)
        glyph.addExtrema()
        glyph.canonicalStart()
        glyph.canonicalContours()
        glyph.round()
    except Exception:
        pass


# ── NORMALIZATION ───────────────────────────────────────

def normalize_glyph(glyph, cp):
    """
    Normalize glyph size and position.

    The glyph is:
        - Scaled to match typography height
        - Centered horizontally
        - Aligned vertically (baseline/descender)

    Args:
        glyph (fontforge.glyph): Glyph to normalize
        cp (int): Unicode codepoint
    """
    bb = glyph.boundingBox()
    xmin, ymin, xmax, ymax = bb

    w = xmax - xmin
    h = ymax - ymin

    if w == 0 or h == 0:
        return

    top, bottom = get_target_metrics(cp)
    scale = (top - bottom) / h

    glyph.transform(psMat.scale(scale))

    bb2 = glyph.boundingBox()
    center_x = (bb2[0] + bb2[2]) / 2

    glyph.transform(psMat.translate(-center_x, bottom - bb2[1]))


# ── SPACING ─────────────────────────────────────────────

def apply_spacing(glyph):
    """
    Apply side bearings and width adjustments.

    Spacing is dynamically determined based on glyph width.

    Args:
        glyph (fontforge.glyph): Glyph to adjust
    """
    bb = glyph.boundingBox()
    w = bb[2] - bb[0]

    if w <= 0:
        glyph.width = 400
        return

    if w < 200:
        left = right = 80
    elif w < 400:
        left = right = 60
    else:
        left = right = 40

    glyph.left_side_bearing = left
    glyph.right_side_bearing = right
    glyph.width = int(w + left + right)


# ── PLACEHOLDER ─────────────────────────────────────────

def draw_underscore_placeholder(glyph, width=400):
    """
    Create an underscore placeholder for missing characters.

    Args:
        glyph (fontforge.glyph): Glyph object to draw into
        width (int): Width of placeholder glyph
    """
    pen = glyph.glyphPen()
    pen.moveTo((40, -120))
    pen.lineTo((width - 40, -120))
    pen.lineTo((width - 40, -50))
    pen.lineTo((40, -50))
    pen.closePath()
    glyph.width = width


# ── MAIN PIPELINE FUNCTION ──────────────────────────────

def build_font(base_folder, folders, output_ttf, font_name, delete_svgs=False):
    """
    Generate a TTF font from SVG glyph datasets.

    Workflow:
        1. Load SVG files from folders
        2. Map filenames to characters
        3. Select best glyph per character
        4. Normalize and clean glyphs
        5. Apply spacing and hinting
        6. Fill missing characters
        7. Generate final TTF file

    Args:
        base_folder (str):
            Root directory containing SVG folders

        folders (list[str]):
            List of subfolder names containing SVG files

        output_ttf (str):
            Path where the generated font will be saved

        font_name (str):
            Name assigned to the font

        delete_svgs (bool, optional):
            If True, deletes SVG folders after generation

    Returns:
        None
    """

    font = fontforge.font()
    font.fontname = font_name
    font.familyname = font_name
    font.fullname = font_name
    font.encoding = "UnicodeFull"
    font.em = UPM
    font.ascent = ASCENDER
    font.descent = abs(DESCENDER)

    char_map = {}

    # Load SVG files
    for folder in folders:
        path = Path(base_folder) / folder
        if not path.exists():
            continue

        for svg in path.glob("*.svg"):
            cp = filename_to_codepoint(svg.name)
            if cp:
                char_map.setdefault(cp, []).append(svg)

    # Select best glyph per character
    for cp, files in char_map.items():
        best_file = None
        best_score = -999

        for f in files:
            temp = fontforge.font()
            g = temp.createChar(cp)

            try:
                g.importOutlines(str(f))
                s = score_glyph(g)
                if s > best_score:
                    best_score = s
                    best_file = f
            except Exception:
                pass

            temp.close()

        if not best_file:
            continue

        glyph = font.createChar(cp)

        try:
            glyph.importOutlines(str(best_file))
            glyph.correctDirection()
            glyph.transform(psMat.scale(0.99))
        except Exception:
            continue

        normalize_glyph(glyph, cp)
        clean_glyph(glyph)
        apply_spacing(glyph)
        glyph.autoHint()

    # Space character
    font.createChar(32).width = 300

    # Fill missing ASCII characters
    for cp in range(33, 127):
        if cp not in char_map:
            g = font.createChar(cp)
            draw_underscore_placeholder(g)
            g.autoHint()

    # Final cleanup
    for g in font.glyphs():
        try:
            g.correctDirection()
        except:
            pass

    font.selection.all()
    font.round()

    Path(output_ttf).parent.mkdir(parents=True, exist_ok=True)
    font.generate(output_ttf)

    # Optional deletion of SVG folders
    if delete_svgs:
        for folder in folders:
            path = Path(base_folder) / folder
            if path.exists():
                shutil.rmtree(path)