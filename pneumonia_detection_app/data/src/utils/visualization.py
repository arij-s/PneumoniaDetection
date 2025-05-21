# utils/visualization.py
import matplotlib.pyplot as plt
from typing import List, Optional
import numpy as np
from config.paths import MODELS_DIR

class Visualizer:
    @staticmethod
    def show_samples(originals: List[np.ndarray], processed: List[np.ndarray], 
                   title: str, n_samples: int = 3):
        """Visualize sample images before and after preprocessing"""
        plt.figure(figsize=(12, 6))
        for i in range(min(n_samples, len(originals))):
            plt.subplot(2, n_samples, i+1)
            plt.imshow(originals[i], cmap='gray')
            plt.title('Original')
            plt.axis('off')
            
            plt.subplot(2, n_samples, i+n_samples+1)
            plt.imshow(processed[i], cmap='gray')
            plt.title('Processed')
            plt.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(MODELS_DIR / f"{title.replace(' ', '_')}.png")
        plt.close()
    @staticmethod
    def visualize_feature_maps(image: np.ndarray, feature_maps: List[np.ndarray], 
                             titles: Optional[List[str]] = None,
                             n_maps: int = 5):
        """
        Visualize feature maps from convolutional layers
        
        Args:
            image: Original image
            feature_maps: List of 2D feature maps
            titles: Titles for each feature map
            n_maps: Number of feature maps to display
        """
        n_maps = min(n_maps, len(feature_maps))
        if titles is None:
            titles = [f"Feature Map {i}" for i in range(n_maps)]
        
        plt.figure(figsize=(15, 8))
        
        # Original image
        plt.subplot(1, n_maps+1, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # Feature maps
        for i in range(n_maps):
            plt.subplot(1, n_maps+1, i+2)
            plt.imshow(feature_maps[i], cmap='viridis')
            plt.title(titles[i])
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(MODELS_DIR / "feature_maps.png")
        plt.close()