# data/preprocessing.py
import cv2
import numpy as np
from joblib import Memory
from config.paths import CACHE_DIR
from config.settings import IMG_SIZE, VISUALIZATION_SIZE

memory = Memory(CACHE_DIR, verbose=0)

class ImagePreprocessor:
    @staticmethod
    @memory.cache
    def preprocess(path, augment=False):
        """Preprocess a single image from path"""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        orig = cv2.resize(img, VISUALIZATION_SIZE)
        img = cv2.resize(img, IMG_SIZE)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        if augment and np.random.rand() > 0.5:
            img = ImagePreprocessor._augment(img)
        
        return orig, img

    @staticmethod
    def _augment(img):
        """Apply random augmentation to image"""
        aug_type = np.random.choice(['flip', 'rotate', 'brightness'])
        if aug_type == 'flip':
            return cv2.flip(img, 1)
        elif aug_type == 'rotate':
            angle = np.random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
            return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        else:
            alpha = np.random.uniform(0.9, 1.1)
            beta = np.random.randint(-5, 5)
            return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)