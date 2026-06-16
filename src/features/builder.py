import numpy as np

from src.features import lbp, glcm, fft_enhanced as fft


class FeatureBuilder:
    def __init__(self, use_lbp=True, use_glcm=True, use_fft=True):

        self.extractors = []

        if use_lbp:
            if hasattr(lbp, "LBPExtractor"):
                self.extractors.append(lbp.LBPExtractor())
            elif hasattr(lbp, "extract_lbp_features"):
                self.extractors.append(lbp.extract_lbp_features)

        if use_glcm:
            if hasattr(glcm, "GLCMExtractor"):
                self.extractors.append(glcm.GLCMExtractor())
            elif hasattr(glcm, "extract_glcm_features"):
                self.extractors.append(glcm.extract_glcm_features)

        if use_fft:
            if hasattr(fft, "FFTExtractor"):
                self.extractors.append(fft.FFTExtractor())
            elif hasattr(fft, "extract_fft_features"):
                self.extractors.append(fft.extract_fft_features)

        if len(self.extractors) == 0:
            raise ValueError("No feature extractors selected!")

    def extract_features(self, image):
        feature_list = []

        for extractor in self.extractors:
            try:
                # Handle BOTH types
                if callable(extractor):
                    features = extractor(image)  # function
                else:
                    features = extractor.extract(image)  # class

                features = np.asarray(features).flatten()
                feature_list.append(features)

            except Exception as e:
                print(f"Error in {extractor}: {e}")

        if len(feature_list) == 0:
            raise ValueError("No features extracted!")

        return np.concatenate(feature_list)