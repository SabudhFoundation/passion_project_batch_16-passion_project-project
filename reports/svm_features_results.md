# Deepfake Detection System — Experimental Results

## 📌 Overview

This project focuses on detecting deepfake images using a feature-based machine learning pipeline combining:

* Local Binary Patterns (LBP)
* Gray Level Co-occurrence Matrix (GLCM)
* Fast Fourier Transform (FFT)
* Statistical Features (mean, standard deviation, skewness, kurtosis)

The model uses an **RBF-approximated SVM** with preprocessing and dimensionality reduction.

---

## ⚙️ Dataset Details

* Training Samples: **100,000**
* Validation Samples: **20,000**
* Test Samples: **20,000**
* Total Features (after enhancement): **171**

---

## 🧪 Feature Engineering

The final feature vector consists of:

* LBP Features: 112 dimensions
* GLCM Features: 40 dimensions
* FFT Features: Remaining dimensions
* Statistical Features: 4 dimensions (mean, std, skewness, kurtosis)

### Preprocessing Steps:

* Energy normalization (FFT)
* Log scaling (FFT)
* Standard normalization (all features)
* Feature concatenation

---

## 🧠 Model Configuration

### Classifier:

* Model: **SVM (SGD + RBF Approximation)**
* Kernel Approximation: `RBFSampler`
* RBF Components: **4000**

### Dimensionality Reduction:

* Method: **PCA**
* Components: **120**

### Hyperparameters:

* Gamma values tested: `[0.005, 0.003, 0.002, 0.0015]`
* Best gamma: **0.005**

---

## 🔍 Threshold Optimization

Instead of default classification threshold (0), a custom threshold was learned using validation data:

* Best Threshold: **-0.0821**

This improved classification performance by aligning decision boundaries with data distribution.

---

## 📊 Results

### Validation Performance:

* Accuracy: **74.64%**
* Best Gamma: **0.005**

---

### Final Test Performance:

| Metric          | Value      |
| --------------- | ---------- |
| Accuracy        | **74.87%** |
| ROC-AUC         | **0.8268** |
| Precision (avg) | 0.75       |
| Recall (avg)    | 0.75       |
| F1-score (avg)  | 0.75       |

---

### Class-wise Performance:

| Class    | Precision | Recall | F1-score |
| -------- | --------- | ------ | -------- |
| Real (0) | 0.79      | 0.68   | 0.73     |
| Fake (1) | 0.72      | 0.82   | 0.77     |

---

## 📈 Key Observations

1. **PCA significantly improved performance**

   * Reduced noise and redundancy
   * Enabled better RBF feature mapping

2. **Statistical features improved ROC-AUC**

   * Captured global distribution properties
   * Enhanced separability

3. **Model bias observed**

   * Higher recall for fake images
   * Lower recall for real images

4. **Threshold tuning improved decision boundary**

   * Better alignment with class distribution

---

## ⚠️ Limitations

* Feature-based approach has limited representation power
* GLCM features are low-dimensional
* Model shows slight bias toward detecting fake images
* Performance ceiling observed around **75–78%**

---

## 🚀 Future Improvements

* Balanced threshold tuning (macro-F1 optimization)
* Feature selection / dimensional pruning
* Ensemble of multiple models (FFT + non-FFT)
* Integration of CNN-based features for higher accuracy

---

## 🏁 Conclusion

The system achieves:

* **~75% accuracy**
* **~0.83 ROC-AUC**

This demonstrates that classical feature-based approaches can effectively detect deepfakes, but further improvements require enhanced feature representations or deep learning integration.

---
