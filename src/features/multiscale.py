import numpy as np
import cv2
import os
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# =========================
# FEATURE EXTRACTOR CLASS
# =========================
class FeatureExtractor:
    def __init__(
        self,
        img_size=128,
        patch_grid=2,
        glcm_distances=[1],
        glcm_angles=[0, np.pi / 2],
        glcm_props=None,
        lbp_radii=[1, 2],
    ):
        self.img_size = img_size
        self.patch_grid = patch_grid
        self.glcm_distances = glcm_distances
        self.glcm_angles = glcm_angles
        self.glcm_props = glcm_props or [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
        ]
        self.lbp_radii = lbp_radii

    def split_into_patches(self, image):
        h, w = image.shape
        ph, pw = h // self.patch_grid, w // self.patch_grid
        return [
            image[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
            for i in range(self.patch_grid)
            for j in range(self.patch_grid)
        ]

    def extract_glcm(self, image):
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        patches = self.split_into_patches(image)

        features = []
        for patch in patches:
            glcm = graycomatrix(
                patch,
                distances=self.glcm_distances,
                angles=self.glcm_angles,
                symmetric=True,
                normed=True,
            )

            for prop in self.glcm_props:
                vals = graycoprops(glcm, prop)
                features.append(np.mean(vals))
                features.append(np.std(vals))

        return np.array(features, dtype=np.float32)

    def extract_lbp(self, image):
        patches = self.split_into_patches(image)
        features = []

        for patch in patches:
            for r in self.lbp_radii:
                n_points = 8 * r

                lbp = local_binary_pattern(patch, n_points, r, method="uniform")

                hist, _ = np.histogram(
                    lbp.ravel(),
                    bins=n_points + 2,
                    range=(0, n_points + 2),
                )

                hist = hist.astype("float32")
                hist /= (hist.sum() + 1e-6)

                features.extend(hist)

        return np.array(features, dtype=np.float32)

    def extract_fft(self, image):
        image = cv2.resize(image, (self.img_size, self.img_size)).astype(np.float32) / 255.0

        h, w = image.shape
        ch, cw = h // 2, w // 2

        hann = np.outer(np.hanning(h), np.hanning(w))
        image = image * hann

        fshift = np.fft.fftshift(np.fft.fft2(image))
        magnitude = np.log1p(np.abs(fshift))

        features = []

        y, x = np.ogrid[:h, :w]
        r = np.hypot(x - cw, y - ch)

        bands = [(0, 8), (8, 16), (16, 30), (30, 45), (45, 64)]

        for r1, r2 in bands:
            mask = (r >= r1) & (r < r2)
            region = magnitude[mask]

            if region.size == 0:
                features.extend([0, 0, 0])
            else:
                features.extend([
                    np.mean(region),
                    np.std(region),
                    np.sum(region ** 2),
                ])

        return np.array(features, dtype=np.float32)

    def extract(self, image, use_glcm=True, use_lbp=True, use_fft=True):
        if image is None:
            return None

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (self.img_size, self.img_size))

        features = []

        if use_glcm:
            features.append(self.extract_glcm(image))

        if use_lbp:
            features.append(self.extract_lbp(image))

        if use_fft:
            features.append(self.extract_fft(image))

        feat = np.concatenate(features)
        return feat / (np.linalg.norm(feat) + 1e-8)


# =========================
# MULTIPROCESS HELPER
# =========================
def process_image(args):
    path, label, extractor = args

    img = cv2.imread(path)
    if img is None:
        return None

    try:
        feat = extractor.extract(img)
        return feat, label
    except Exception:
        return None


# =========================
# DATASET PROCESSING
# =========================
def process_dataset(dataset_path, extractor):
    all_tasks = []

    for label_name, label_val in [("real", 0), ("fake", 1)]:
        folder = os.path.join(dataset_path, label_name)

        if not os.path.isdir(folder):
            continue

        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        for f in files:
            all_tasks.append((f, label_val, extractor))

    with Pool(cpu_count()) as pool:
        results = list(pool.imap(process_image, all_tasks))

    X, y = [], []
    for res in results:
        if res is not None:
            feat, label = res
            X.append(feat)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# =========================
# SAVE FUNCTION
# =========================
def save_features(dataset_path, output_file, extractor):
    X, y = process_dataset(dataset_path, extractor)

    if X.size == 0:
        return None

    with open(output_file, "wb") as f:
        pickle.dump((X, y), f)

    return X.shape