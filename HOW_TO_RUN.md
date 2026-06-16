# How to Run — Deepfake Detection API

## Prerequisites

- Python 3.10 or 3.11
- pip

---

## 1. Install Dependencies

Run from the project root:

```bash
pip install -r requirements.txt
```

**Minimal install** (SVM and MLP methods only — no deep learning frameworks needed):

```bash
pip install numpy opencv-python scikit-learn scikit-image scipy joblib Pillow \
            fastapi "uvicorn[standard]" python-multipart pydantic
```

---

## 2. Place Model Weights

Trained weights are not included in the repository. Place them at the paths shown below before starting the server. Missing weights return HTTP 503 for that specific method — the server still starts and other methods remain available.

```
models/
├── svm/
│   ├── svm_fft.pkl
│   ├── svm_features.pkl
│   └── normalizer_features.pkl
├── mlp/
│   ├── glcm.pkl
│   ├── lbp.pkl
│   ├── fft.pkl
│   └── combined.pkl
├── cnn/
│   ├── resnet50_finetuned.keras
│   ├── inceptionv3_finetuned.keras
│   └── efficientnet_finetuned.keras
└── dual_channel/
    ├── dual_channel_inception.pth
    └── dual_channel_resnet.pth
```

---

## 3. Start the API Server

Run from the project root (the `python -m` prefix ensures the correct Python environment is used):

```bash
python -m uvicorn src.inference.api:app --reload
```

The server starts at `http://localhost:8000`.  
Interactive docs (Swagger UI): `http://localhost:8000/docs`

To use a different port:

```bash
python -m uvicorn src.inference.api:app --reload --port 8080
```

---

## 4. Test the Endpoints

### Health check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{ "status": "ok" }
```

### List all methods

```bash
curl http://localhost:8000/methods
```

Expected response:

```json
{
  "methods": [
    "lbp_svm", "fft_svm", "combined_svm",
    "glcm_mlp", "lbp_mlp", "fft_mlp", "combined_mlp",
    "resnet50", "inceptionv3", "efficientnet",
    "dual_channel_inception", "dual_channel_resnet"
  ]
}
```

### Classify an image

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/face.jpg" \
  -F "method=fft_svm"
```

Expected response:

```json
{
  "method": "fft_svm",
  "label": "real",
  "prediction": 0,
  "confidence": 0.81,
  "reported_accuracy": 68.34
}
```

`prediction` is `0` for real, `1` for fake.  
`confidence` is the model's probability for the predicted label (0–1).

---

## 5. Method Reference

| Method key | Features | Classifier | Test Accuracy | Requires |
|---|---|---|---|---|
| `lbp_svm` | LBP | SVM | 65.11% | scikit-learn |
| `fft_svm` | FFT | SVM | 68.34% | scikit-learn |
| `combined_svm` | LBP + GLCM + FFT | SVM | 74.87% | scikit-learn |
| `glcm_mlp` | GLCM | MLP | 66.89% | scikit-learn |
| `lbp_mlp` | LBP | MLP | 79.93% | scikit-learn |
| `fft_mlp` | FFT | MLP | 62.32% | scikit-learn |
| `combined_mlp` | LBP + GLCM + FFT | MLP | 82.33% | scikit-learn |
| `resnet50` | CNN | Fine-tuned ResNet50 | 71.80% | tensorflow |
| `inceptionv3` | CNN | Fine-tuned InceptionV3 | 87.63% | tensorflow |
| `efficientnet` | CNN | Fine-tuned EfficientNetB0 | 94.41% | tensorflow |
| `dual_channel_inception` | Spatial + FFT | Dual-channel (InceptionV3) | 98.92% | torch |
| `dual_channel_resnet` | Spatial + FFT | Dual-channel (ResNet50) | 98.92% | torch |

---

## 6. Gradio Web Interface

A browser-based UI is available as an alternative to calling the API directly.

```bash
python interface/app.py
```

Opens at `http://localhost:7860`.  
Upload a face image on the left, choose a method from the dropdown, click **Classify** — the label, confidence, method, and reported accuracy appear on the right.

---

## 7. Train the Models

Run each script from the project root. Weights are saved to `models/` automatically.

```bash
# Classical ML — SVM
python -m src.training.train_svm_fft
python -m src.training.train_svm_features

# Classical ML — MLP
python -m src.training.train_mlp

# CNN (requires TensorFlow)
python -m src.training.train_resnet
python -m src.training.train_inception
python -m src.training.train_efficientnet

# Dual-channel (requires PyTorch)
python -m src.training.train_dual_channel --backbone inception
python -m src.training.train_dual_channel --backbone resnet
```

Training scripts expect the dataset at:

```
deepfake_dataset/real-vs-fake/
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

## 7. Use the Predictor Directly (no server)

```python
from src.inference.predict import DeepfakePredictor

predictor = DeepfakePredictor()
result = predictor.predict("path/to/face.jpg", method="fft_svm")
print(result)
# {'label': 'real', 'confidence': 0.81, 'prediction': 0}
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `HTTP 503` | Weight file missing for that method | Place the `.pkl` / `.keras` / `.pth` file at the expected path (see section 2) |
| `HTTP 422` | Image file could not be read | Check the file is a valid JPEG/PNG and not corrupted |
| `HTTP 400` | Unknown method name | Use one of the 12 keys listed in section 5 |
| `RuntimeError: Form data requires python-multipart` | Package missing from Python env | `pip install python-multipart` in the same Python used to run uvicorn |
| `PCA is expecting N features` | Inference extractor doesn't match training | Ensure `src/features/builder.py` imports `fft_enhanced as fft` (not `fft`) |
