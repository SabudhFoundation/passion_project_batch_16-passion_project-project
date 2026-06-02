"""Glyph vectorization for FontForge."""

import os
import re
import cv2
import vtracer
import tempfile


def rescale_svg_for_fontforge(svg_content, canvas_w=500, canvas_h=700):
    """Rescale SVG coordinates to a 1000×1000 font space."""
    if "viewBox" not in svg_content:
        w_match = re.search(r'<svg[^>]*\swidth="([^"]+)"', svg_content)
        h_match = re.search(r'<svg[^>]*\sheight="([^"]+)"', svg_content)

        view_w = canvas_w
        view_h = canvas_h

        if w_match:
            try:
                view_w = float(re.sub(r"[^\d.]", "", w_match.group(1)))
            except ValueError:
                pass

        if h_match:
            try:
                view_h = float(re.sub(r"[^\d.]", "", h_match.group(1)))
            except ValueError:
                pass

        svg_content = re.sub(
            r"(<svg\b)",
            f'\\1 viewBox="0 0 {view_w} {view_h}"',
            svg_content,
            count=1,
        )
        canvas_w = view_w
        canvas_h = view_h

    else:
        vb_match = re.search(
            r'viewBox="[^"]*?\s+([\d.]+)\s+([\d.]+)"',
            svg_content,
        )
        if vb_match:
            canvas_w = float(vb_match.group(1))
            canvas_h = float(vb_match.group(2))

    svg_content = re.sub(
        r'(<svg[^>]*\s)width="[^"]*"',
        r'\g<1>width="1000"',
        svg_content,
    )
    svg_content = re.sub(
        r'(<svg[^>]*\s)height="[^"]*"',
        r'\g<1>height="1000"',
        svg_content,
    )

    sx = 1000.0 / canvas_w if canvas_w else 1.0
    sy = 1000.0 / canvas_h if canvas_h else 1.0

    svg_content = re.sub(
        r"(<svg\b[^>]*>)",
        r'\1' + f'<g transform="scale({sx:.6f},{sy:.6f})">',
        svg_content,
        count=1,
    )
    svg_content = re.sub(r"(</svg\s*>)", r"</g>\1", svg_content, count=1)

    return svg_content


def vectorize_image(input_path, output_path):
    """Convert a raster glyph image to SVG using VTracer."""
    try:
        vtracer.convert_image_to_svg_py(
            input_path,
            output_path,
            colormode="binary",
            mode="spline",
            filter_speckle=1,
            corner_threshold=35,
            length_threshold=1.5,
            max_iterations=20,
            splice_threshold=20,
            path_precision=4,
        )
    except Exception as e:
        raise RuntimeError(f"VTracer error for {input_path}: {e}")


def process_vectorization(input_image, canvas_w=500, canvas_h=700):
    """Convert a processed glyph image into a FontForge-ready SVG."""
    if input_image is None:
        raise ValueError("input_image is None")

    if len(input_image.shape) == 3:
        img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    else:
        img = input_image.copy()

    img = cv2.GaussianBlur(img, (3, 3), 0)

    temp_img_path = None
    temp_svg_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
            temp_img_path = temp_img.name

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as temp_svg:
            temp_svg_path = temp_svg.name

        if not cv2.imwrite(temp_img_path, img):
            raise RuntimeError(f"cv2.imwrite failed for {temp_img_path}")

        vectorize_image(temp_img_path, temp_svg_path)

        with open(temp_svg_path, "r", encoding="utf-8") as file:
            svg_content = file.read()

        if not svg_content.strip():
            raise RuntimeError("VTracer produced an empty SVG")

        return rescale_svg_for_fontforge(
            svg_content,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
        )

    finally:
        for path in (temp_img_path, temp_svg_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
