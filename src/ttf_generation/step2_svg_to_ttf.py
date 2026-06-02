"""FontForge helper for the SVG to TTF stage."""

import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from ttf_final import build_font


svg_folder = os.environ.get("FF_SVG_FOLDER", "output_svgs")
folders_raw = os.environ.get("FF_FOLDERS", "kid2,kid15")
output_ttf = os.environ.get("FF_OUTPUT_TTF", "final_font/font.ttf")
font_name = os.environ.get("FF_FONT_NAME", "MyCustomFont")
delete_svgs = os.environ.get("FF_DELETE_SVGS", "0") == "1"

folders = [folder.strip() for folder in folders_raw.split(",") if folder.strip()]

PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
svg_folder_abs = os.path.join(PROJECT_DIR, svg_folder)
output_ttf_abs = os.path.join(PROJECT_DIR, output_ttf)

print(f"[Stage 2] SVG folder : {svg_folder_abs}")
print(f"[Stage 2] Folders    : {folders}")
print(f"[Stage 2] Output TTF : {output_ttf_abs}")

total_svgs = 0

for folder in folders:
    folder_path = os.path.join(svg_folder_abs, folder)

    if not os.path.isdir(folder_path):
        print(f"[Stage 2] Missing folder: {folder_path}")
        continue

    svg_count = len([
        name for name in os.listdir(folder_path)
        if name.lower().endswith(".svg")
    ])
    total_svgs += svg_count
    print(f"[Stage 2] {folder} → {svg_count} SVGs")

if total_svgs == 0:
    print("[Stage 2] No SVGs found.")
    sys.exit(1)

try:
    build_font(
        base_folder=svg_folder_abs,
        folders=folders,
        output_ttf=output_ttf_abs,
        font_name=font_name,
        delete_svgs=delete_svgs,
    )
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)

if not os.path.exists(output_ttf_abs):
    print(f"[Stage 2] TTF was not created: {output_ttf_abs}")
    sys.exit(1)

print(f"[Stage 2] Font saved → {output_ttf_abs}")
