# Deepfake Image Detection

A deepfake face-image classifier built across eight experiments, progressing from classical handcrafted features to a dual-channel spatial+frequency deep learning architecture. The project ships a **FastAPI endpoint** that accepts an image and a method name and returns a real/fake classification from any of the 12 trained pipelines.

---

## Experiment Results

| # | Method | Features | Classifier | Test Accuracy |
|---|--------|----------|------------|--------------|
| 1 | `lbp_svm` | LBP | SVM (RBF approx) | 65.11% |
| 2 | `fft_svm` | FFT | SVM (RBF approx) | 68.34% |
| 3 | `combined_svm` | LBP + GLCM + FFT | SVM (RBF approx) | 74.87% |
| 4a | `glcm_mlp` | GLCM | MLP | 66.89% |
| 4b | `lbp_mlp` | LBP | MLP | 79.93% |
| 4c | `fft_mlp` | FFT | MLP | 62.32% |
| 4d | `combined_mlp` | LBP + GLCM + FFT | MLP | **82.33%** |
| 5 | `resnet50` | CNN (ResNet50) | Fine-tuned sigmoid | 71.80% |
| 6 | `inceptionv3` | CNN (InceptionV3) | Fine-tuned sigmoid | 87.63% |
| 7 | `efficientnet` | CNN (EfficientNetB0) | Fine-tuned sigmoid | 94.41% |
| 8a | `dual_channel_inception` | InceptionV3 spatial + FFT freq | Dual-channel (PyTorch) | **98.92%** |
| 8b | `dual_channel_resnet` | ResNet50 spatial + FFT freq | Dual-channel (PyTorch) | **98.92%** |

---

## Project Structure

```
passion-project-deep-fake-identifiers/
│
├── src/
│   ├── features/               # Feature extraction (shared across training and inference)
│   │   ├── lbp.py              # Local Binary Patterns → 10-dim histogram
│   │   ├── glcm.py             # Gray-Level Co-occurrence Matrix → 4-dim vector
│   │   ├── fft.py              # FFT frequency-domain features (Hann window)
│   │   ├── fft_enhanced.py     # Extended FFT pipeline
│   │   ├── multiscale.py       # Multi-scale feature extractor
│   │   ├── deep_features.py    # EfficientNet CNN embedding extractor
│   │   └── builder.py          # FeatureBuilder — combines any subset of the above
│   │
│   ├── models/                 # Architecture definitions (no trained weights here)
│   │   ├── svm_classifier.py   # SVMClassifier: PCA → RBFSampler → SGDClassifier + threshold
│   │   ├── normalizer.py       # FeatureNormalizer: per-group LBP/GLCM/FFT normalization
│   │   └── dual_channel/
│   │       ├── detector.py     # DualChannelDetector (PyTorch nn.Module)
│   │       ├── spatial_channel.py
│   │       └── frequency_channel.py
│   │
│   ├── training/               # Scripts that produce trained model artifacts
│   │   ├── train_svm_glcm.py
│   │   ├── train_svm_fft.py
│   │   ├── train_svm_features.py
│   │   ├── train_mlp.py
│   │   ├── train_efficientnet.py
│   │   ├── train_inception.py
│   │   ├── train_resnet.py
│   │   ├── train_dual_channel.py
│   │   └── trainer.py
│   │
│   └── inference/
│       ├── predict.py          # DeepfakePredictor — lazy-loading, all 12 methods
│       └── api.py              # FastAPI application
│
├── models/                     # Trained artifacts only (excluded from git)
│   ├── svm/
│   │   ├── svm_fft.pkl
│   │   ├── svm_features.pkl
│   │   └── normalizer_features.pkl
│   ├── mlp/                    # Populated after running train_mlp.py
│   │   ├── glcm.pkl
│   │   ├── lbp.pkl
│   │   ├── fft.pkl
│   │   └── combined.pkl
│   ├── cnn/                    # Populated after running train_*.py CNN scripts
│   │   ├── resnet50_finetuned.keras
│   │   ├── inceptionv3_finetuned.keras
│   │   └── efficientnet_finetuned.keras
│   ├── dual_channel/           # Populated after running train_dual_channel.py
│   │   ├── dual_channel_inception.pth
│   │   └── dual_channel_resnet.pth
│   └── configs/                # JSON metadata (best params, label maps)
│
├── data/
│   ├── raw/                    # Original images (gitignored)
│   └── features/
│       └── combined/           # Cached feature pkl files (gitignored)
│
├── scripts/
│   └── explain/                # Grad-CAM visualisation demos (exploratory)
│       ├── gradcam_efficientnet.py
│       ├── gradcam_inception.py
│       └── gradcam_resnet50.py
│
├── reports/
│   └── figures/
│
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Setup

**Python 3.11**

```bash
pip install -r requirements.txt
```

SVM and MLP methods only need `scikit-learn`, `opencv-python`, and `scikit-image`.  
CNN methods additionally need `tensorflow`.  
Dual-channel methods additionally need `torch` and `torchvision`.

---

## Dataset

Not included due to size. Place it at:

```
deepfake_dataset/
└── real-vs-fake/
    ├── train/
    │   ├── real/
    │   └── fake/
    ├── valid/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/
```

---

## Training

Run each script from the project root:

```bash
# Classical ML
python -m src.training.train_svm_fft
python -m src.training.train_svm_features
python -m src.training.train_mlp

# CNN (TensorFlow / Keras)
python -m src.training.train_resnet
python -m src.training.train_inception
python -m src.training.train_efficientnet

# Dual-channel (PyTorch)
python -m src.training.train_dual_channel --backbone inception
python -m src.training.train_dual_channel --backbone resnet
```

Trained weights are saved to `models/` and are gitignored.

---

## FastAPI Inference Endpoint

### Start the server

```bash
uvicorn src.inference.api:app --reload
```

Interactive docs: `http://localhost:8000/docs`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/methods` | List all 12 method keys |
| POST | `/predict` | Classify an image |

### POST /predict

Accepts `multipart/form-data`:

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | Image (JPEG, PNG, …) |
| `method` | string | One of the 12 method keys |

**Example (curl):**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@face.jpg" \
  -F "method=combined_svm"
```

**Response:**

```json
{
  "method": "combined_svm",
  "label": "fake",
  "prediction": 1,
  "confidence": 0.83,
  "reported_accuracy": 74.87
}
```

**Method not yet trained (weights file missing):**

```json
HTTP 503
{ "detail": "Model 'efficientnet' weights not found at: models/cnn/efficientnet_finetuned.keras" }
```

### Programmatic use

```python
from src.inference.predict import DeepfakePredictor

predictor = DeepfakePredictor()
result = predictor.predict("face.jpg", method="dual_channel_inception")
# {"label": "fake", "confidence": 0.97, "prediction": 1}
```

---

## Feature Details

| Feature | Implementation | Output dim |
|---------|----------------|-----------|
| LBP | Uniform LBP P=8, R=1, 128×128 grayscale | 10 |
| GLCM | Contrast, energy, homogeneity, correlation (distance=1, angle=0) | 4 |
| FFT | Hann-windowed 2D FFT, log-magnitude + phase bands | 157 |
| Combined | Concatenation of all three, group-normalized | 171 |

---

## Author

Liyakat Hussain
