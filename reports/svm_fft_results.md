# 📊 Deepfake Detection using FFT + SVM (RBF Approx)

## 🔍 Overview

This project focuses on detecting deepfake images using frequency-domain features extracted via **Fast Fourier Transform (FFT)** and a scalable nonlinear classifier using **RBF Kernel Approximation + SGDClassifier**.

---

## 🧠 Feature Engineering

### 1. FFT Features

* Extracted frequency-domain representation of images
* Captures:

  * Global structural inconsistencies
  * High-frequency artifacts
* Feature size: **145 dimensions**

---

### 2. Preprocessing Pipeline

#### ✅ Energy Normalization

* Formula:
  [
  X = \frac{X}{\sum |X|}
  ]
* Purpose:

  * Removes scale/brightness variation
  * Normalizes FFT magnitude

---

#### ✅ Log Transformation

* Applied:
  [
  \log(1 + |X|)
  ]
* Purpose:

  * Reduces skewness
  * Stabilizes large FFT values

---

#### ✅ Standard Scaling

* Mean = 0, Std = 1
* Ensures stable optimization

---

## 🤖 Model Architecture

### 🔹 RBF Kernel Approximation

* Used: `RBFSampler`
* Converts nonlinear problem → linear space
* Components:

  * **n_components = 4000**

---

### 🔹 Classifier

* Model: `SGDClassifier`
* Loss: `hinge` (SVM-like)
* Regularization:

  * **alpha = 1e-4**
* Max iterations: 2000

---

### 🔹 Dimensionality Reduction

* Method: **PCA**
* Components: **100**

---

## ⚙️ Hyperparameter Tuning

### Tuned Parameter:

* **gamma (RBF kernel)**

### Values Tested:

```text
0.02, 0.01, 0.005
```

### Best Value:

```text
gamma = 0.01
```

---

## 📊 Dataset

| Split      | Samples |
| ---------- | ------- |
| Train      | 100,000 |
| Validation | 20,000  |
| Test       | 20,000  |

* Balanced dataset:

  * Real: 50%
  * Fake: 50%

---

## 📈 Results

### 🔹 Validation Performance

* Accuracy: **68.55%**

---

### 🔹 Final Test Performance

| Metric    | Class 0 (Real) | Class 1 (Fake) |
| --------- | -------------- | -------------- |
| Precision | 0.67           | 0.70           |
| Recall    | 0.72           | 0.64           |
| F1-score  | 0.70           | 0.67           |

### ✅ Overall Accuracy:

```text
68.34%
```

---

## 🧠 Key Observations

### ✔ Strengths

* Stable model (no overfitting)
* Balanced predictions across classes
* Efficient training (~15–20 sec)

---

### ❗ Limitations

* FFT captures global patterns only
* Weak at detecting:

  * Texture inconsistencies
  * Local artifacts

---

## 🚀 Insights

* Model performance plateaued around:

```text
68–70%
```

* Indicates:

```text
Feature limitation, not model limitation
```

---

## 🔥 Future Improvements

### 1. Add GLCM Features (Texture)

* Expected gain: **+5–8%**

---

### 2. CNN Feature Extraction (ResNet)

* Expected accuracy:

```text
80–90%
```

---

### 3. Hybrid Features

* Combine:

  * FFT + GLCM + CNN
* Best performance potential

---

## 🏁 Conclusion

* Successfully built a scalable deepfake detection pipeline using:

  * Frequency-domain features
  * Nonlinear SVM approximation

* Achieved:

```text
~68% accuracy using FFT alone
```

* Identified next steps for significant improvement through feature enhancement.

---
