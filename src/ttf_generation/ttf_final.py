"""Build a TTF font from SVG glyphs."""

import shutil
from pathlib import Path

import fontforge
import psMat


UPM = 1000
ASCENDER = 800
DESCENDER = -200

CAP_HEIGHT = 700
ASCENDER_HEIGHT = 750
X_HEIGHT = 500
DESCENDER_DEPTH = -200

ASCENDER_CHARS = set("bdfhklt")
DESCENDER_CHARS = set("gjpqy")
X_HEIGHT_CHARS = set("aceimnorsuvwxz")


def get_target_metrics(cp):
    """Return the target top and bottom for a glyph."""
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


def filename_to_codepoint(filename):
    """Map a glyph filename to a Unicode codepoint."""
    stem = Path(filename).stem.lower()
    parts = stem.split("_")

    if parts[0] == "capital" and len(parts) > 1 and len(parts[1]) == 1:
        return ord(parts[1].upper())

    if parts[0] == "small" and len(parts) > 1 and len(parts[1]) == 1:
        return ord(parts[1].lower())

    if len(stem) == 1 and stem.isalpha():
        return ord(stem)

    if len(stem) == 1 and stem.isdigit():
        return ord(stem)

    return None


def score_glyph(glyph):
    """Score a glyph candidate and prefer cleaner shapes."""
    bb = glyph.boundingBox()
    if bb == (0, 0, 0, 0):
        return -999.0

    xmin, ymin, xmax, ymax = bb
    w = xmax - xmin
    h = ymax - ymin

    if w <= 0 or h <= 0:
        return -999.0

    contour_count = 0

    try:
        for _contour in glyph.foreground:
            contour_count += 1
    except Exception:
        contour_count = 1

    aspect_ratio = w / h
    area = w * h

    return (
        -abs(aspect_ratio - 0.6) * 2.0
        + area / 100000.0
        - abs(contour_count - 2) * 0.3
    )


def clean_glyph(glyph):
    """Apply light cleanup to reduce jagged outlines."""
    operations = [
        lambda g: g.removeOverlap(),
        lambda g: g.simplify(2.0),
        lambda g: g.addExtrema(),
        lambda g: g.canonicalStart(),
        lambda g: g.canonicalContours(),
        lambda g: g.round(),
    ]

    for operation in operations:
        try:
            operation(glyph)
        except Exception:
            pass


def normalize_glyph(glyph, cp):
    """Scale and place a glyph using the target font metrics."""
    bb = glyph.boundingBox()
    xmin, ymin, xmax, ymax = bb

    w = xmax - xmin
    h = ymax - ymin

    if w <= 0 or h <= 0:
        return

    top, bottom = get_target_metrics(cp)
    target_h = top - bottom

    if target_h <= 0:
        return

    scale = target_h / h
    glyph.transform(psMat.scale(scale))

    bb = glyph.boundingBox()
    center_x = (bb[0] + bb[2]) / 2.0
    translate_y = bottom - bb[1]

    glyph.transform(psMat.translate(-center_x, translate_y))


def apply_spacing(glyph):
    """Apply simple side bearings and advance width."""
    bb = glyph.boundingBox()
    width = bb[2] - bb[0]

    if width <= 0:
        glyph.width = 500
        return

    if width < 250:
        left = right = 80
    elif width < 500:
        left = right = 60
    else:
        left = right = 40

    glyph.left_side_bearing = left
    glyph.right_side_bearing = right
    glyph.width = int(width + left + right)


def draw_underscore_placeholder(glyph, width=500):
    """Draw a simple underscore placeholder."""
    pen = glyph.glyphPen()
    pen.moveTo((50, -150))
    pen.lineTo((width - 50, -150))
    pen.lineTo((width - 50, -80))
    pen.lineTo((50, -80))
    pen.closePath()
    glyph.width = width


def build_font(base_folder, folders, output_ttf, font_name, delete_svgs=False):
    """Generate a TTF font from SVG glyph folders."""
    font = fontforge.font()
    font.fontname = font_name
    font.familyname = font_name
    font.fullname = font_name
    font.encoding = "UnicodeFull"
    font.em = UPM
    font.ascent = ASCENDER
    font.descent = abs(DESCENDER)

    font.os2_typoascent = ASCENDER
    font.os2_typodescent = DESCENDER
    font.os2_typolinegap = 0
    font.os2_winascent = ASCENDER
    font.os2_windescent = abs(DESCENDER)
    font.hhea_ascent = ASCENDER
    font.hhea_descent = DESCENDER
    font.hhea_linegap = 0

    char_map = {}

    for folder in folders:
        path = Path(base_folder) / folder
        if not path.exists():
            print(f"[TTF] Missing folder: {path}")
            continue

        for svg in sorted(path.glob("*.svg")):
            cp = filename_to_codepoint(svg.name)
            if cp is not None:
                char_map.setdefault(cp, []).append(svg)

    if not char_map:
        print("[TTF] No SVG glyphs found.")
        return

    success_count = 0

    for cp, files in sorted(char_map.items()):
        best_file = None
        best_score = -999.0

        for file_path in files:
            temp_font = fontforge.font()
            temp_glyph = temp_font.createChar(cp)

            try:
                temp_glyph.importOutlines(str(file_path))
                score = score_glyph(temp_glyph)
                if score > best_score:
                    best_score = score
                    best_file = file_path
            except Exception as e:
                print(f"[TTF] Score failed for {file_path.name}: {e}")
            finally:
                temp_font.close()

        if best_file is None:
            continue

        glyph = font.createChar(cp)

        try:
            glyph.importOutlines(str(best_file))
        except Exception as e:
            print(f"[TTF] Import failed for {best_file.name}: {e}")
            continue

        if glyph.boundingBox() == (0, 0, 0, 0):
            continue

        try:
            glyph.correctDirection()
            normalize_glyph(glyph, cp)
            clean_glyph(glyph)
            apply_spacing(glyph)
            glyph.autoHint()
            success_count += 1
        except Exception as e:
            print(f"[TTF] Post-processing failed for {best_file.name}: {e}")
            success_count += 1

    print(f"[TTF] Imported {success_count}/{len(char_map)} glyphs")

    space = font.createChar(32)
    space.width = 250

    for cp in range(33, 127):
        if cp not in char_map:
            glyph = font.createChar(cp)
            draw_underscore_placeholder(glyph)
            glyph.autoHint()

    for glyph in font.glyphs():
        try:
            glyph.correctDirection()
        except Exception:
            pass

    font.selection.all()
    font.round()

    out_path = Path(output_ttf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    font.generate(str(out_path))
    print(f"[TTF] Font saved → {out_path}")

    if delete_svgs:
        for folder in folders:
            path = Path(base_folder) / folder
            if path.exists():
                shutil.rmtree(path)
