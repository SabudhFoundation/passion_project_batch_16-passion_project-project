[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=23178844&assignment_repo_type=AssignmentRepo)

# Fontaneous : Few-Shot Grapheme Extrapolation for Typeface Generation

## Overview

This project implements an end-to-end system that converts handwritten text images into a **custom TTF font file**.

It covers the complete pipeline:

```

Input Images
↓
Preprocessing (denoise, binarization, line removal)
↓
Segmentation (character extraction)
↓
OCR Labeling (character classification)
↓
Scoring + Grouping
↓
User Selection (best character per class)
↓
Dataset Creation
↓
Vectorization (SVG generation)
↓
TTF Font Generation (FontForge)

```



## Python Version

```
Python 3.12.3
```



## Installation

### System dependencies (WSL / Ubuntu required)

```bash
sudo apt-get update
sudo apt-get install -y fontforge python3-fontforge
```

### Python dependencies

```bash
pip install -r require.txt
```

### Run application

```bash
streamlit run src/main.py
```


## Docker Deployment
###### (Go to dedicated docker directory)

### Build the image:

```bash
docker build -t fontaneouse .
```

### Run with Docker Compose:

```bash
docker compose up --build
```


### The application will run in localhost:8501 or < your ip address>:8501"



## Project Structure

```
├── README.md
│
├── reports
│   └── figures/                       # pipeline visuals (preprocess → segmentation → OCR)
│
├── docker/
|      |──DockerFile/                            # Docker-related scripts
|      |──docker-compose.yaml
|      └──setup_linux_venv.sh
|       
├── src
│   │
│   ├── preprocessing_data
│   │   └── preprocessing.py          # image cleaning + normalization
│   │
│   ├── segmentation
│   │   ├── Segmentation.py           # character segmentation logic
│   │   └── tesseract/                # (moved from root for clarity)
│   │
│   ├── char_ocr_labeling
│   │   ├── cli_selection.py         # manual/interactive selection
│   │   ├── config.py                # OCR + pipeline configs
│   │   ├── labeling.py              # label assignment logic
│   │   ├── ocr_engine.py            # OCR inference layer
│   │   ├── normalization.py         # character normalization
│   │   ├── scoring.py               # confidence/quality scoring
│   │   ├── utils.py                 # shared helpers
│   │   ├── saver.py                 # dataset persistence layer
│   │   └── main.py                  # OCR pipeline entry point
│   │
│   ├── dataset
│   │   ├── final_dataset/           # structured labeled dataset
│   │   ├── best_chars/              # selected clean samples
│   │   └── best_chars_inverted/     # inverted variants
│   │
│   ├── vectorization
│   │   ├── preprocessing.py         # vectorization preprocessing stage
│   │   └── vectorization.py         # raster → vector conversion (SVG)
│   │
│   ├── font_generation
│   │   ├── step2_svg_to_ttf.py      # SVG → font glyph mapping
│   │   └── ttf_final.py             # final TTF builder/export
│   │
│   └── main.py                      # Streamlit / pipeline controller
│
├── require.txt
├── system-requirements.txt
├── .gitignore
└── LICENSE

```



## Pipeline Explanation

### 1. Preprocessing

* Grayscale conversion
* Noise removal
* Adaptive thresholding
* Morphological cleaning
* Line removal using inpainting

Output: clean binary glyph image



### 2. Segmentation

Extracts individual characters from word/image using:

* connected components
* bounding box filtering
* morphological validation

Output: list of character crops



### 3. OCR Labeling

Uses EasyOCR to assign character labels:

* filters invalid predictions
* confidence thresholding
* single-character enforcement

Output:

```
{ img, label, confidence, score }
```



### 4. Scoring System

Each character is ranked using:

* OCR confidence
* fill ratio
* connected components
* aspect ratio
* centroid alignment

Used to select best glyph per class.



### 5. User Selection

A grid-based UI (or CLI fallback):

* groups by label
* sorts by score
* user selects best candidate per class

Output:

```
data[sub]["best"]
```



### 6. Dataset Creation

Generates structured dataset:

```
final_dataset/
   ├── student_1/
   │     ├── capital_a/images/
   │     ├── small_b/images/
```

Also saves:

* best_chars/
* best_chars_inverted/



### 7. Vectorization

Each glyph is converted:

```
PNG → SVG (VTracer)
```

Stored per label for font generation.



### 8. Font Generation

Using FontForge:

* imports SVG glyphs
* normalizes metrics
* applies spacing rules
* handles missing characters
* exports `.ttf`

Output:

```
new_font.ttf
```



## Key Features

* End-to-end handwriting-to-font system
* Robust OCR + scoring pipeline
* Human-in-the-loop selection
* SVG-based vectorization
* Fully automated TTF generation
* Modular and extensible architecture



## Requirements

```
opencv-python
numpy
easyocr
streamlit
fontforge
vtracer
```



## Run Pipeline

### Streamlit UI

```bash
streamlit run src/main.py
```



## Outputs

* `final_dataset/` → full labeled dataset
* `best_chars/` → selected glyphs
* `best_chars_inverted/` → inverted dataset
* `new_font.ttf` → final font

