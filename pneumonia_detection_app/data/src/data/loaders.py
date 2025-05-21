# data/loaders.py
from pathlib import Path
from joblib import Parallel, delayed
from typing import Tuple, List
import numpy as np
from .preprocessing import ImagePreprocessor

class DataLoader:
    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor or ImagePreprocessor()

    def load_from_folder(self, folder: Path, augment: bool = False, max_images: int = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load and preprocess images from folder"""
        paths = [f for f in folder.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        
        if max_images:
            paths = paths[:max_images]
        
        print(f"Loading {len(paths)} images from {folder}...")
        
        results = Parallel(n_jobs=-1)(
            delayed(self.preprocessor.preprocess)(p, augment) 
            for p in paths
        )
        
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            return [], []
            
        return zip(*valid_results)