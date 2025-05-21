import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.utils import resample
from skimage.exposure import is_low_contrast
from typing import Dict
import random

class DataValidator:
    def __init__(self):
        self.stats = {
            'duplicates_removed': 0,
            'low_quality_removed': 0,
            'corrupted_removed': 0,
            'images_balanced': 0
        }

    def _is_low_quality(self, img: np.ndarray) -> bool:
        if img.mean() < 20 or img.mean() > 235:
            return True
        return is_low_contrast(img, fraction_threshold=0.05)

    def _compute_image_hash(self, img: np.ndarray) -> int:
        small_img = cv2.resize(img, (16, 16))
        diff = small_img[1:, :] > small_img[:-1, :]
        return sum([2**i for i, v in enumerate(diff.flatten()) if v])

class DataCleaner(DataValidator):
    def __init__(self, raw_dir: Path, cleaned_dir: Path):
        super().__init__()
        self.raw_dir = raw_dir
        self.cleaned_dir = cleaned_dir
        self._setup_dirs()

    def _setup_dirs(self):
        for subset in ['train', 'test']:
            for class_ in ['PNEUMONIA', 'NORMAL']:
                (self.cleaned_dir / subset / class_).mkdir(parents=True, exist_ok=True)

    def clean_dataset(self) -> Dict[str, int]:
        self._remove_duplicates()
        self._filter_low_quality()
        self._balance_classes()
        return self.stats

    def _remove_duplicates(self, tolerance: int = 5):
        hashes = {}
        for img_path in tqdm(list(self.raw_dir.glob('**/*.jp*g')) + 
                             list(self.raw_dir.glob('**/*.png')),
                             desc="Removing duplicates"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                self.stats['corrupted_removed'] += 1
                continue

            img_hash = self._compute_image_hash(img)
            duplicate_found = any(abs(img_hash - h) <= tolerance for h in hashes)
            if duplicate_found:
                self.stats['duplicates_removed'] += 1
            else:
                hashes[img_hash] = img_path

    def _filter_low_quality(self):
        for img_path in tqdm(list(self.raw_dir.glob('**/*.jp*g')) + 
                             list(self.raw_dir.glob('**/*.png')),
                             desc="Filtering low-quality"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            if self._is_low_quality(img):
                self.stats['low_quality_removed'] += 1
            else:
                self._save_clean_image(img, img_path)

    def _save_clean_image(self, img: np.ndarray, src_path: Path):
        relative_path = src_path.relative_to(self.raw_dir)
        dest_path = self.cleaned_dir / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest_path), img)

    def _balance_classes(self):
        """
        Ensure equal number of NORMAL and PNEUMONIA images in each subset (train/test).
        This downsamples the majority class.
        """
        for subset in ['train', 'test']:
            norm_dir = self.cleaned_dir / subset / 'NORMAL'
            pneu_dir = self.cleaned_dir / subset / 'PNEUMONIA'

            norm_imgs = list(norm_dir.glob('*'))
            pneu_imgs = list(pneu_dir.glob('*'))

            min_len = min(len(norm_imgs), len(pneu_imgs))

            if len(norm_imgs) > min_len:
                to_remove = random.sample(norm_imgs, len(norm_imgs) - min_len)
            elif len(pneu_imgs) > min_len:
                to_remove = random.sample(pneu_imgs, len(pneu_imgs) - min_len)
            else:
                continue

            for path in to_remove:
                path.unlink()
                self.stats['images_balanced'] += 1
