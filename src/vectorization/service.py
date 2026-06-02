import sys
from pathlib import Path

from vectorization.preprocessing import process_glyph_image
from vectorization.vectorization import process_vectorization


def generate_svgs(data, font_name, svg_root):
    svg_folder = Path(svg_root) / font_name
    svg_folder.mkdir(parents=True, exist_ok=True)

    items = []
    for sub in data:
        if "best" in data[sub]:
            items.extend(data[sub]["best"])

    total = len(items)
    saved = 0

    for i, item in enumerate(items, 1):
        label = item["label"]
        img = item["img"]

        print(f"\r   Processing {i}/{total} → {label}", end="")
        sys.stdout.flush()

        glyph = process_glyph_image(img, f"{label}.png")
        if glyph is None:
            continue

        svg = process_vectorization(glyph)

        path = svg_folder / f"{label}.svg"
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)

        saved += 1

    print()
    print(f"[SVG] Generated {saved}/{total}")

    return svg_folder