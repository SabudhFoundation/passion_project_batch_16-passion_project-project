import pickle
import time
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


class SVMClassifier:
    """
    Scalable SVM Classifier with:
    - Optional PCA (dimensionality reduction)
    - RBF kernel approximation
    - Threshold tuning for better accuracy
    """

    def __init__(
        self,
        model_type="rbf_approx",
        C=1.0,
        gamma=0.01,
        class_weight="balanced",
        use_pca=True,
        pca_components=120,
        rbf_components=4000,
        random_state=42
    ):
        self.model_type = model_type
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.rbf_components = rbf_components
        self.random_state = random_state

        self.best_threshold = 0.0
        self.model = self._build_pipeline()

    # =====================================
    # BUILD PIPELINE
    # =====================================
    def _build_pipeline(self):
        steps = []

        # PCA (optional)
        if self.use_pca:
            steps.append((
                "pca",
                PCA(
                    n_components=self.pca_components,
                    svd_solver="randomized",
                    random_state=self.random_state
                )
            ))

        # RBF Approximation
        if self.model_type == "rbf_approx":
            steps.append((
                "rbf",
                RBFSampler(
                    gamma=self.gamma,
                    n_components=self.rbf_components,
                    random_state=self.random_state
                )
            ))

        # Final classifier
        steps.append(("clf", self._build_classifier()))

        return Pipeline(steps)

    # =====================================
    # CLASSIFIER SELECTION
    # =====================================
    def _build_classifier(self):

        if self.model_type == "linear":
            return LinearSVC(
                C=self.C,
                class_weight=self.class_weight,
                max_iter=3000,
                tol=1e-3
            )

        elif self.model_type in ["sgd", "rbf_approx"]:
            return SGDClassifier(
                loss="hinge",
                alpha=3e-5,
                max_iter=4000,
                tol=1e-4,
                learning_rate="optimal",
                class_weight=self.class_weight,
                n_jobs=-1,
                random_state=self.random_state
            )

        else:
            raise ValueError("Invalid model_type")

    # =====================================
    # TRAIN
    # =====================================
    def train(self, X, y):
        print(f"\nTraining {self.model_type.upper()} model...")
        start = time.time()

        self.model.fit(X, y)

        print(f"Done in {time.time() - start:.2f} sec")

    # =====================================
    # DECISION SCORES
    # =====================================
    def decision_scores(self, X):
        return self.model.decision_function(X)

    # =====================================
    # THRESHOLD TUNING
    # =====================================
    def tune_threshold(self, X, y):
        scores = self.decision_scores(X)

        thresholds = np.linspace(scores.min(), scores.max(), 200)

        best_acc = 0
        best_t = 0

        for t in thresholds:
            preds = (scores > t).astype(int)
            acc = accuracy_score(y, preds)

            if acc > best_acc:
                best_acc = acc
                best_t = t

        self.best_threshold = best_t

        print(f"Best Threshold: {best_t:.4f}")
        print(f"Tuned Accuracy: {best_acc:.4f}")

        return best_t, best_acc

    # =====================================
    # PREDICT
    # =====================================
    def predict(self, X):
        scores = self.decision_scores(X)
        return (scores > self.best_threshold).astype(int)

    # =====================================
    # EVALUATE
    # =====================================
    def evaluate(self, X, y):
        y_pred = self.predict(X)

        acc = accuracy_score(y, y_pred)

        print(f"\nAccuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred))

        # ROC-AUC (optional)
        try:
            scores = self.decision_scores(X)
            auc = roc_auc_score(y, scores)
            print(f"\nROC-AUC: {auc:.4f}")
        except Exception:
            print("ROC-AUC not available")

        return acc

    # =====================================
    # SAVE / LOAD
    # =====================================
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "threshold": self.best_threshold
            }, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.best_threshold = data.get("threshold", 0.0)