import os
import shutil
import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytesseract
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel


SEGMENTATION_DIR = os.path.dirname(os.path.abspath(__file__))


def get_tesseract_cmd():
    """Return the Tesseract command for the current platform."""
    env_cmd = os.environ.get("TESSERACT_CMD")

    if env_cmd:
        return env_cmd

    bundled_cmd = os.path.join(
        SEGMENTATION_DIR,
        "Tesseract-OCR",
        "tesseract.exe",
    )

    if os.name == "nt" and os.path.isfile(bundled_cmd):
        return bundled_cmd

    command_path = shutil.which("tesseract")

    if command_path:
        return command_path

    return "tesseract"


def prepare_input_images(
    input_image,
    clahe_clip_limit,
    clahe_tile_grid_size,
    adaptive_block_size,
    adaptive_c,
    upscale_factor,
):
    if input_image is None:
        raise ValueError("input_image is None")

    img_bgr = input_image.copy()
    if upscale_factor > 1.0:
        img_bgr = cv2.resize(
            img_bgr,
            None,
            fx=upscale_factor,
            fy=upscale_factor,
            interpolation=cv2.INTER_CUBIC,
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=clahe_tile_grid_size,
    )
    enhanced_image = clahe.apply(gray_image)

    binary_image = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        adaptive_block_size,
        adaptive_c,
    )

    return {
        "img_rgb": img_rgb,
        "enhanced_image": enhanced_image,
        "binary_image": binary_image,
        "image_height": gray_image.shape[0],
        "image_width": gray_image.shape[1],
    }


def load_vit_components(vit_model_name):
    vit_processor = ViTImageProcessor.from_pretrained(vit_model_name)
    vit_model = ViTModel.from_pretrained(
        vit_model_name,
        output_attentions=True,
        add_pooling_layer=False,
        attn_implementation="eager",
    )
    vit_model.eval()
    return vit_processor, vit_model


def build_attention_rollout(attentions, discard_ratio, head_fusion):
    token_count = attentions[0].size(-1)
    result = torch.eye(token_count)

    for attention_layer in attentions:
        attention_heads = attention_layer.squeeze(0)

        if head_fusion == "mean":
            fused_attention = attention_heads.mean(0)
        elif head_fusion == "max":
            fused_attention = attention_heads.max(0).values
        else:
            fused_attention = attention_heads[head_fusion]

        flat_attention = fused_attention.view(-1)
        threshold = flat_attention.quantile(discard_ratio)
        fused_attention = fused_attention.clone()
        fused_attention[fused_attention < threshold] = 0
        fused_attention = fused_attention + torch.eye(token_count)
        fused_attention = fused_attention / fused_attention.sum(-1, keepdim=True)
        result = torch.matmul(fused_attention, result)

    return result


def calculate_vit_patch_scores(
    rgb_image,
    vit_processor,
    vit_model,
    discard_ratio,
    head_fusion,
):
    image_height, image_width = rgb_image.shape[:2]
    pil_image = Image.fromarray(rgb_image)
    model_input = vit_processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        model_output = vit_model(**model_input, output_attentions=True)

    rollout = build_attention_rollout(
        model_output.attentions,
        discard_ratio,
        head_fusion,
    )

    cls_attention = rollout[0, 1:].cpu().numpy()
    side = int(math.isqrt(cls_attention.shape[0]))
    patch_scores = cls_attention.reshape(side, side).astype(np.float32)
    patch_scores = (patch_scores - patch_scores.min()) / (
        patch_scores.max() - patch_scores.min() + 1e-8
    )

    attention_map = cv2.resize(
        patch_scores,
        (image_width, image_height),
        interpolation=cv2.INTER_CUBIC,
    )

    return patch_scores, attention_map


def detect_line_bands(
    binary_image,
    image_height,
    line_threshold,
    min_line_height,
    line_padding,
):
    pixel_row = binary_image.mean(axis=1)
    line_bands = []
    in_line = False
    start_y = 0

    for y in range(image_height):
        if not in_line and pixel_row[y] > line_threshold:
            in_line = True
            start_y = y
        elif in_line and pixel_row[y] <= line_threshold:
            in_line = False
            if (y - start_y) >= min_line_height:
                line_bands.append(
                    (
                        max(0, start_y - line_padding),
                        min(image_height, y + line_padding),
                    )
                )

    if in_line and (image_height - start_y) >= min_line_height:
        line_bands.append((max(0, start_y - line_padding), image_height))

    return line_bands


def find_word_ranges_in_line(
    binary_image,
    line_top,
    line_bottom,
    image_width,
    merge_kernel_size,
    word_gap_threshold,
    word_gap_min_width,
    min_word_width,
):
    kernel = np.ones(merge_kernel_size) / merge_kernel_size
    line_strip = binary_image[line_top:line_bottom, :]
    column_projection = line_strip.mean(axis=0)
    smoothed_projection = np.convolve(column_projection, kernel, mode="same")
    above_threshold = smoothed_projection > word_gap_threshold

    runs = []
    in_run = False
    run_start = 0

    for x in range(image_width):
        if not in_run and above_threshold[x]:
            in_run = True
            run_start = x
        elif in_run and not above_threshold[x]:
            in_run = False
            runs.append((run_start, x))

    if in_run:
        runs.append((run_start, image_width))

    if not runs:
        return []

    merged_runs = [list(runs[0])]
    for start_x, end_x in runs[1:]:
        if (start_x - merged_runs[-1][1]) < word_gap_min_width:
            merged_runs[-1][1] = end_x
        else:
            merged_runs.append([start_x, end_x])

    return [
        (start_x, end_x)
        for start_x, end_x in merged_runs
        if (end_x - start_x) >= min_word_width
    ]


def detect_lines_and_words(
    binary_image,
    image_height,
    image_width,
    line_threshold,
    min_line_height,
    line_padding,
    word_padding,
    merge_kernel_size,
    word_gap_threshold,
    word_gap_min_width,
    min_word_width,
    max_black_ratio,
):
    line_bands = detect_line_bands(
        binary_image,
        image_height,
        line_threshold,
        min_line_height,
        line_padding,
    )

    all_words = []
    for line_index, (line_top, line_bottom) in enumerate(line_bands):
        word_counter = 0
        word_ranges = find_word_ranges_in_line(
            binary_image,
            line_top,
            line_bottom,
            image_width,
            merge_kernel_size,
            word_gap_threshold,
            word_gap_min_width,
            min_word_width,
        )

        for word_left, word_right in word_ranges:
            x0 = max(0, word_left - word_padding)
            y0 = max(0, line_top - word_padding)
            x1 = min(image_width, word_right + word_padding)
            y1 = min(image_height, line_bottom + word_padding)

            word_region = binary_image[y0:y1, x0:x1]
            total_pixels = word_region.size
            black_pixels = np.sum(word_region == 0)
            black_ratio = black_pixels / total_pixels

            if black_ratio > max_black_ratio:
                continue

            all_words.append((line_index, word_counter, x0, y0, x1, y1))
            word_counter += 1

    return line_bands, all_words


def read_character_boxes(
    gray_image,
    ocr_scale_factor,
    min_ocr_dim,
    psm_mode,
):
    if gray_image.size == 0:
        return []

    scaled = cv2.resize(
        gray_image,
        None,
        fx=ocr_scale_factor,
        fy=ocr_scale_factor,
        interpolation=cv2.INTER_CUBIC,
    )
    scaled_height = scaled.shape[0]

    try:
        raw_boxes = pytesseract.image_to_boxes(
            scaled,
            config=f"--psm {psm_mode} --oem 1",
        )
    except Exception as error:
        command = pytesseract.pytesseract.tesseract_cmd
        raise RuntimeError(
            f"Tesseract OCR failed for command '{command}': {error}"
        ) from error

    characters = []

    for line in raw_boxes.strip().split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue

        character = parts[0]
        left, bottom, right, top = (
            int(parts[1]),
            int(parts[2]),
            int(parts[3]),
            int(parts[4]),
        )

        char_x0 = left // ocr_scale_factor
        char_x1 = right // ocr_scale_factor
        char_y0 = (scaled_height - top) // ocr_scale_factor
        char_y1 = (scaled_height - bottom) // ocr_scale_factor

        if (char_x1 - char_x0) >= min_ocr_dim and (char_y1 - char_y0) >= min_ocr_dim:
            if character.strip():
                characters.append((character, char_x0, char_y0, char_x1, char_y1))

    return characters


def build_padded_region(
    x0,
    y0,
    x1,
    y1,
    char_padding,
    image_width,
    image_height,
):
    region_x0 = max(0, x0 - char_padding)
    region_y0 = max(0, y0 - char_padding)
    region_x1 = min(image_width, max(region_x0 + 1, x1 + char_padding))
    region_y1 = min(image_height, max(region_y0 + 1, y1 + char_padding))
    return region_x0, region_y0, region_x1, region_y1


def run_vit_on_word_crops(
    all_words,
    img_rgb,
    vit_processor,
    vit_model,
    discard_ratio,
    head_fusion,
):
    word_attention = {}

    for line_index, word_index, x0, y0, x1, y1 in all_words:
        crop = img_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        patch_scores, attention_map = calculate_vit_patch_scores(
            crop,
            vit_processor,
            vit_model,
            discard_ratio,
            head_fusion,
        )
        word_attention[(line_index, word_index)] = {
            "patch_scores": patch_scores,
            "attention_map": attention_map,
        }

    return word_attention


def run_ocr_on_word_crops(
    all_words,
    enhanced_image,
    ocr_scale_factor,
    min_ocr_dim,
):
    word_characters = {}

    for line_index, word_index, x0, y0, x1, y1 in all_words:
        word_key = (line_index, word_index)
        crop_gray = enhanced_image[y0:y1, x0:x1]

        word_characters[word_key] = read_character_boxes(
            crop_gray,
            ocr_scale_factor,
            min_ocr_dim,
            8,
        )

    return word_characters


def build_ocr_character_regions(
    all_words,
    word_characters,
    char_padding,
    image_width,
    image_height,
):
    character_regions = []

    for line_index, word_index, x0, y0, x1, y1 in all_words:
        word_key = (line_index, word_index)
        characters = word_characters.get(word_key, [])

        for char_index, (character, char_x0, char_y0, char_x1, char_y1) in enumerate(characters):
            region_x0, region_y0, region_x1, region_y1 = build_padded_region(
                x0 + char_x0,
                y0 + char_y0,
                x0 + char_x1,
                y0 + char_y1,
                char_padding,
                image_width,
                image_height,
            )

            character_regions.append(
                {
                    "source": "ocr",
                    "character": character,
                    "line_index": line_index,
                    "word_index": word_index,
                    "char_index": char_index,
                    "x0": region_x0,
                    "y0": region_y0,
                    "x1": region_x1,
                    "y1": region_y1,
                }
            )

    return character_regions


def build_global_character_regions(
    global_characters,
    char_padding,
    image_width,
    image_height,
):
    character_regions = []

    for char_index, (character, char_x0, char_y0, char_x1, char_y1) in enumerate(global_characters):
        region_x0, region_y0, region_x1, region_y1 = build_padded_region(
            char_x0,
            char_y0,
            char_x1,
            char_y1,
            char_padding,
            image_width,
            image_height,
        )

        character_regions.append(
            {
                "source": "global_ocr",
                "character": character,
                "char_index": char_index,
                "x0": region_x0,
                "y0": region_y0,
                "x1": region_x1,
                "y1": region_y1,
            }
        )

    return character_regions


def run_ocr_segmentation_branch(
    binary_image,
    enhanced_image,
    image_height,
    image_width,
    line_threshold,
    min_line_height,
    line_padding,
    word_padding,
    merge_kernel_size,
    word_gap_threshold,
    word_gap_min_width,
    min_word_width,
    max_black_ratio,
    ocr_scale_factor,
    min_ocr_dim,
    char_padding,
):
    _, all_words = detect_lines_and_words(
        binary_image,
        image_height,
        image_width,
        line_threshold,
        min_line_height,
        line_padding,
        word_padding,
        merge_kernel_size,
        word_gap_threshold,
        word_gap_min_width,
        min_word_width,
        max_black_ratio,
    )

    word_characters = run_ocr_on_word_crops(
        all_words,
        enhanced_image,
        ocr_scale_factor,
        min_ocr_dim,
    )

    return build_ocr_character_regions(
        all_words,
        word_characters,
        char_padding,
        image_width,
        image_height,
    )


def run_global_ocr_branch(
    enhanced_image,
    image_height,
    image_width,
    global_ocr_scale_factor,
    min_ocr_dim,
    char_padding,
):
    global_characters = read_character_boxes(
        enhanced_image,
        global_ocr_scale_factor,
        min_ocr_dim,
        11,
    )

    return build_global_character_regions(
        global_characters,
        char_padding,
        image_width,
        image_height,
    )


def calculate_character_overlap(first_region, second_region):
    overlap_left = max(first_region["x0"], second_region["x0"])
    overlap_top = max(first_region["y0"], second_region["y0"])
    overlap_right = min(first_region["x1"], second_region["x1"])
    overlap_bottom = min(first_region["y1"], second_region["y1"])

    overlap_width = max(0, overlap_right - overlap_left)
    overlap_height = max(0, overlap_bottom - overlap_top)
    overlap_area = overlap_width * overlap_height

    if overlap_area <= 0:
        return 0.0

    first_area = max(1, (first_region["x1"] - first_region["x0"]) * (first_region["y1"] - first_region["y0"]))
    second_area = max(
        1,
        (second_region["x1"] - second_region["x0"]) * (second_region["y1"] - second_region["y0"]),
    )
    return overlap_area / min(first_area, second_area)


def combine_character_regions(
    ocr_character_regions,
    global_character_regions,
    character_overlap_threshold,
):
    combined_regions = [dict(region) for region in ocr_character_regions]

    for global_region in global_character_regions:
        best_index = -1
        best_overlap = 0.0

        for index, combined_region in enumerate(combined_regions):
            overlap = calculate_character_overlap(combined_region, global_region)
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index

        if best_index == -1 or best_overlap < character_overlap_threshold:
            combined_regions.append(dict(global_region))
            continue

        combined_regions[best_index]["x0"] = min(
            combined_regions[best_index]["x0"],
            global_region["x0"],
        )
        combined_regions[best_index]["y0"] = min(
            combined_regions[best_index]["y0"],
            global_region["y0"],
        )
        combined_regions[best_index]["x1"] = max(
            combined_regions[best_index]["x1"],
            global_region["x1"],
        )
        combined_regions[best_index]["y1"] = max(
            combined_regions[best_index]["y1"],
            global_region["y1"],
        )

        if combined_regions[best_index]["source"] != global_region["source"]:
            combined_regions[best_index]["source"] = "combined"

    combined_regions.sort(
        key=lambda region: (
            region["y0"],
            region["x0"],
            region.get("line_index", 0),
            region.get("word_index", 0),
            region.get("char_index", 0),
        )
    )
    return combined_regions


def calculate_attention_score_for_region(
    word_patch_scores,
    crop_height,
    crop_width,
    region_x0,
    region_y0,
    region_x1,
    region_y1,
):
    patch_rows, patch_cols = word_patch_scores.shape
    grid_x0 = max(0, int(region_x0 / crop_width * patch_cols))
    grid_y0 = max(0, int(region_y0 / crop_height * patch_rows))
    grid_x1 = min(patch_cols, max(grid_x0 + 1, int(region_x1 / crop_width * patch_cols)))
    grid_y1 = min(patch_rows, max(grid_y0 + 1, int(region_y1 / crop_height * patch_rows)))
    return float(word_patch_scores[grid_y0:grid_y1, grid_x0:grid_x1].mean())


def extract_dominant_character_crop(
    rgb_image,
    binary_image,
    word_patch_scores,
    char_x0,
    char_y0,
    char_x1,
    char_y1,
    min_cc_area,
):
    crop_height, crop_width = rgb_image.shape[:2]

    char_x0 = max(0, char_x0)
    char_y0 = max(0, char_y0)
    char_x1 = min(crop_width, char_x1)
    char_y1 = min(crop_height, char_y1)

    if char_x1 <= char_x0 or char_y1 <= char_y0:
        return None

    rgb_crop = rgb_image[char_y0:char_y1, char_x0:char_x1].copy()
    binary_crop = binary_image[char_y0:char_y1, char_x0:char_x1]

    ink = (binary_crop > 128).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink,
        connectivity=8,
    )

    if num_labels <= 1:
        return rgb_crop

    best_label = -1
    best_score = -1.0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_cc_area:
            continue

        box_x0 = int(stats[label, cv2.CC_STAT_LEFT])
        box_y0 = int(stats[label, cv2.CC_STAT_TOP])
        box_width = int(stats[label, cv2.CC_STAT_WIDTH])
        box_height = int(stats[label, cv2.CC_STAT_HEIGHT])

        attention_score = calculate_attention_score_for_region(
            word_patch_scores,
            crop_height,
            crop_width,
            char_x0 + box_x0,
            char_y0 + box_y0,
            char_x0 + box_x0 + box_width,
            char_y0 + box_y0 + box_height,
        )
        combined_score = area * attention_score

        if combined_score > best_score:
            best_score = combined_score
            best_label = label

    if best_label == -1:
        return rgb_crop

    keep_mask = labels == best_label
    remove_mask = (ink > 0) & ~keep_mask
    rgb_crop[remove_mask] = [0, 0, 0]
    return rgb_crop


def build_character_crops(
    character_regions,
    img_rgb,
    binary_image,
    global_patch_scores,
    min_cc_area,
):
    character_crops = []

    for character_region in character_regions:
        masked_crop = extract_dominant_character_crop(
            img_rgb,
            binary_image,
            global_patch_scores,
            character_region["x0"],
            character_region["y0"],
            character_region["x1"],
            character_region["y1"],
            min_cc_area,
        )

        if masked_crop is not None:
            character_crops.append(masked_crop)

    return character_crops


def process_segmentation(input_image):
    tesseract_cmd = get_tesseract_cmd()

    vit_model_name = "google/vit-base-patch16-224"
    discard_ratio = 0.9
    head_fusion = "mean"

    upscale_factor = 2.0

    clahe_clip_limit = 3.0
    clahe_tile_grid_size = (8, 8)
    adaptive_block_size = 31
    adaptive_c = -5

    line_threshold = 20.0
    min_line_height = 10
    line_padding = 4

    word_gap_threshold = 8.0
    word_gap_min_width = 6
    min_word_width = 8
    merge_kernel_size = 5
    word_padding = 18

    character_overlap_threshold = 0.60

    ocr_scale_factor = 6
    global_ocr_scale_factor = 1
    min_ocr_dim = 3
    char_padding = 9

    min_cc_area = 12
    max_black_ratio = 0.90

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    prepared_images = prepare_input_images(
        input_image,
        clahe_clip_limit,
        clahe_tile_grid_size,
        adaptive_block_size,
        adaptive_c,
        upscale_factor,
    )

    vit_processor, vit_model = load_vit_components(vit_model_name)

    with ThreadPoolExecutor(max_workers=2) as executor:
        ocr_future = executor.submit(
            run_ocr_segmentation_branch,
            prepared_images["binary_image"],
            prepared_images["enhanced_image"],
            prepared_images["image_height"],
            prepared_images["image_width"],
            line_threshold,
            min_line_height,
            line_padding,
            word_padding,
            merge_kernel_size,
            word_gap_threshold,
            word_gap_min_width,
            min_word_width,
            max_black_ratio,
            ocr_scale_factor,
            min_ocr_dim,
            char_padding,
        )
        global_ocr_future = executor.submit(
            run_global_ocr_branch,
            prepared_images["enhanced_image"],
            prepared_images["image_height"],
            prepared_images["image_width"],
            global_ocr_scale_factor,
            min_ocr_dim,
            char_padding,
        )
        global_patch_scores, _ = calculate_vit_patch_scores(
            prepared_images["img_rgb"],
            vit_processor,
            vit_model,
            discard_ratio,
            head_fusion,
        )

        ocr_character_regions = ocr_future.result()
        global_character_regions = global_ocr_future.result()

    combined_character_regions = combine_character_regions(
        ocr_character_regions,
        global_character_regions,
        character_overlap_threshold,
    )

    character_crops = build_character_crops(
        combined_character_regions,
        prepared_images["img_rgb"],
        prepared_images["binary_image"],
        global_patch_scores,
        min_cc_area,
    )

    return character_crops


def process_segemntation(input_image):
    return process_segmentation(input_image)
