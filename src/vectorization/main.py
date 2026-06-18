import cv2, os
from preprocessing import process_glyph_image
from vectorization import process_vectorization


def process_folder(input_folder="ocr_output", output_folder="output"):
    for root, _, files in os.walk(input_folder):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                out_dir = os.path.join(output_folder, os.path.relpath(root, input_folder))
                os.makedirs(out_dir, exist_ok=True)

                img = cv2.imread(os.path.join(root, f), 0)
                p = process_glyph_image(img, f)

                if p is not None:
                    cv2.imwrite(os.path.join(out_dir, f), p)
                    open(os.path.join(out_dir, os.path.splitext(f)[0] + ".svg"), "w").write(process_vectorization(p))
                    print("✅", f)

if __name__ == "__main__":
    process_folder(output_folder="output")                    