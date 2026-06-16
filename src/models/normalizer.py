import numpy as np
import pickle


class FeatureNormalizer:
    """
    Feature Normalizer for [LBP | GLCM | FFT]

    Features:
    - Independent normalization per feature group
    - FFT preprocessing (energy normalization + log scaling)
    - Optional feature weighting
    """

    def __init__(
        self,
        lbp_dim,
        glcm_dim,
        use_log_fft=True,
        use_energy_norm=True,
        fft_weight=1.0,
        glcm_weight=1.0,
        lbp_scale=True
    ):
        self.lbp_dim = lbp_dim
        self.glcm_dim = glcm_dim

        self.use_log_fft = use_log_fft
        self.use_energy_norm = use_energy_norm

        self.fft_weight = fft_weight
        self.glcm_weight = glcm_weight
        self.lbp_scale = lbp_scale

        self.params = {}

    # =====================================
    # SPLIT FEATURES
    # =====================================
    def split_features(self, X):
        if X.ndim != 2:
            raise ValueError("Input must be 2D array [samples, features]")

        lbp = X[:, :self.lbp_dim]
        glcm_end = self.lbp_dim + self.glcm_dim
        glcm = X[:, self.lbp_dim:glcm_end]
        fft = X[:, glcm_end:]

        return lbp, glcm, fft

    # =====================================
    # LBP
    # =====================================
    def fit_lbp(self, lbp):
        if not self.lbp_scale:
            return

        self.params["lbp_mean"] = np.mean(lbp, axis=0)
        self.params["lbp_std"] = np.std(lbp, axis=0) + 1e-8

    def transform_lbp(self, lbp):
        if not self.lbp_scale:
            return lbp

        return (lbp - self.params["lbp_mean"]) / self.params["lbp_std"]

    # =====================================
    # GLCM
    # =====================================
    def fit_glcm(self, glcm):
        self.params["glcm_mean"] = np.mean(glcm, axis=0)
        self.params["glcm_std"] = np.std(glcm, axis=0) + 1e-8

    def transform_glcm(self, glcm):
        glcm = (glcm - self.params["glcm_mean"]) / self.params["glcm_std"]
        return glcm * self.glcm_weight

    # =====================================
    # FFT PROCESSING
    # =====================================
    def process_fft(self, fft):
        # Energy normalization (per sample)
        if self.use_energy_norm:
            energy = np.sum(np.abs(fft), axis=1, keepdims=True) + 1e-8
            fft = fft / energy

        # Log scaling
        if self.use_log_fft:
            fft = np.log1p(np.abs(fft))

        return fft

    # =====================================
    # FFT
    # =====================================
    def fit_fft(self, fft):
        fft = self.process_fft(fft)

        self.params["fft_mean"] = np.mean(fft, axis=0)
        self.params["fft_std"] = np.std(fft, axis=0) + 1e-8

    def transform_fft(self, fft):
        fft = self.process_fft(fft)
        fft = (fft - self.params["fft_mean"]) / self.params["fft_std"]

        return fft * self.fft_weight

    # =====================================
    # MAIN FIT
    # =====================================
    def fit(self, X):
        lbp, glcm, fft = self.split_features(X)

        self.fit_lbp(lbp)
        self.fit_glcm(glcm)
        self.fit_fft(fft)

        return self   # enables chaining

    # =====================================
    # MAIN TRANSFORM
    # =====================================
    def transform(self, X):
        lbp, glcm, fft = self.split_features(X)

        lbp = self.transform_lbp(lbp)
        glcm = self.transform_glcm(glcm)
        fft = self.transform_fft(fft)

        return np.concatenate([lbp, glcm, fft], axis=1)

    # =====================================
    # FIT + TRANSFORM
    # =====================================
    def fit_transform(self, X):
        return self.fit(X).transform(X)

    # =====================================
    # SAVE / LOAD
    # =====================================
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.__dict__.update(pickle.load(f))