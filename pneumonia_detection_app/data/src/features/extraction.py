# features/extraction.py
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from joblib import Memory
try:
    # For local development
    from config.paths import CACHE_DIR
except ImportError:
    # For production (Render)
    from pneumonia_detection_app.data.src.config.paths import CACHE_DIR

memory = Memory(CACHE_DIR, verbose=0)

class FeatureExtractor:
    @staticmethod
    @memory.cache
    def extract(img: np.ndarray) -> np.ndarray:
        """Extract features from a single image"""
        # Histogram features
        hist = cv2.calcHist([img], [0], None, [16], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        
        # LBP features
        lbp = local_binary_pattern(img, 8, 1, method='default')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 16))
        lbp_hist = lbp_hist.astype(np.float32)
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
        
        # HOG features
        h = hog(img, orientations=8, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1), visualize=False, feature_vector=True)
        
        # Statistical features
        mean, std = img.mean(), img.std()
        
        return np.concatenate([hist, lbp_hist, h[:32], [mean, std]])