#!/usr/bin/env python3
"""
Main pipeline for pneumonia detection with integrated data cleaning and reporting
"""

import numpy as np
from time import time
from pathlib import Path
from joblib import Parallel, delayed, dump
from typing import Dict, List
import os
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend that doesn't use Tkinter
os.environ['OMP_NUM_THREADS'] = '1'  # Disable OpenMP parallelism
os.environ['MKL_NUM_THREADS'] = '1'  # Disable MKL parallelism

# Configuration and Paths
from config.paths import (
    RAW_DATA_DIR,
    CLEANED_DATA_DIR,
    MODELS_DIR,
    TRAIN_POS,
    TRAIN_NEG,
    TEST_POS,
    TEST_NEG
)

# Core Components
from data.loaders import DataLoader
from data.cleaning import DataCleaner
from features.extraction import FeatureExtractor
from models.training import ModelTrainer
from models.evaluation import ModelEvaluator

# Reporting and Visualization
from utils.quality import DataQualityReport, ModelPerformanceReport
from utils.visualization import Visualizer


def run_data_cleaning(clean_data: bool) -> Dict[str, Path]:
    if not clean_data:
        return {
            'train_pos': TRAIN_POS,
            'train_neg': TRAIN_NEG,
            'test_pos': TEST_POS,
            'test_neg': TEST_NEG
        }

    print("\n=== DATA CLEANING PHASE ===")
    cleaner = DataCleaner(raw_dir=RAW_DATA_DIR, cleaned_dir=CLEANED_DATA_DIR)
    cleaning_stats = cleaner.clean_dataset()
    reporter = DataQualityReport(output_dir=MODELS_DIR / "reports")

    def count_images(path: Path) -> int:
        return len(list(path.glob('*.jp*g'))) + len(list(path.glob('*.png')))

    before_counts = {
        'train/PNEUMONIA': count_images(RAW_DATA_DIR / "train/PNEUMONIA"),
        'train/NORMAL': count_images(RAW_DATA_DIR / "train/NORMAL"),
        'test/PNEUMONIA': count_images(RAW_DATA_DIR / "test/PNEUMONIA"),
        'test/NORMAL': count_images(RAW_DATA_DIR / "test/NORMAL")
    }

    after_counts = {
        'train/PNEUMONIA': count_images(CLEANED_DATA_DIR / "train/PNEUMONIA"),
        'train/NORMAL': count_images(CLEANED_DATA_DIR / "train/NORMAL"),
        'test/PNEUMONIA': count_images(CLEANED_DATA_DIR / "test/PNEUMONIA"),
        'test/NORMAL': count_images(CLEANED_DATA_DIR / "test/NORMAL")
    }

    reporter.generate_visual_report(cleaning_stats, before_counts, after_counts)
    reporter.generate_text_report(cleaning_stats, before_counts, after_counts)

    return {
        'train_pos': CLEANED_DATA_DIR / "train/PNEUMONIA",
        'train_neg': CLEANED_DATA_DIR / "train/NORMAL",
        'test_pos': CLEANED_DATA_DIR / "test/PNEUMONIA",
        'test_neg': CLEANED_DATA_DIR / "test/NORMAL"
    }


def run_training_pipeline(data_paths: Dict[str, Path], debug: bool = False) -> List[Dict]:
    print("\n=== MODEL TRAINING PHASE ===")
    max_images = 100 if debug else None

    data_loader = DataLoader()
    feature_extractor = FeatureExtractor()
    model_trainer = ModelTrainer()

    print("\nLoading training data...")
    pos_orig, pos_proc = data_loader.load_from_folder(
        data_paths['train_pos'], augment=True, max_images=max_images
    )
    neg_orig, neg_proc = data_loader.load_from_folder(
        data_paths['train_neg'], augment=True, max_images=max_images
    )

    if pos_proc:
        Visualizer.show_samples(pos_orig, pos_proc, 'Pneumonia Samples')
    if neg_proc:
        Visualizer.show_samples(neg_orig, neg_proc, 'Normal Samples')

    print("\nLoading test data...")
    te_pos_orig, te_pos_proc = data_loader.load_from_folder(
        data_paths['test_pos'], max_images=max_images
    )
    te_neg_orig, te_neg_proc = data_loader.load_from_folder(
        data_paths['test_neg'], max_images=max_images
    )

    print("\nExtracting features...")
    def batch_extract(images: List) -> np.ndarray:
        return np.vstack(
            Parallel(n_jobs=-1)(
                delayed(feature_extractor.extract)(img) for img in images
            )
        )

    X_train = np.vstack([
        batch_extract(pos_proc),
        batch_extract(neg_proc)
    ])
    y_train = np.array([1] * len(pos_proc) + [0] * len(neg_proc))

    X_test = np.vstack([
        batch_extract(te_pos_proc),
        batch_extract(te_neg_proc)
    ])
    y_test = np.array([1] * len(te_pos_proc) + [0] * len(te_neg_proc))

    print(f"\nFeature shape: {X_train.shape[1]} features per image")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    trained_models = model_trainer.tune_and_train(X_train, y_train, n_iter=5)

    results = []
    for name, model in trained_models.items():
        eval_results = ModelEvaluator.evaluate(name, model, X_test, y_test)
        eval_results['model_object'] = model  # ✅ Include model for saving later
        results.append(eval_results)

    return results


def generate_final_report(results: List[Dict], runtime: float):
    reporter = ModelPerformanceReport(output_dir=MODELS_DIR / "reports")
    reporter.generate_model_performance_report(results)

    best = max(results, key=lambda x: x['recall'])

    # ✅ Save the best model
    dump(best['model_object'], MODELS_DIR / "best_model.joblib")

    with open(MODELS_DIR / "reports" / "pipeline_summary.txt", "w") as f:
        f.write(f"Pipeline completed in {runtime:.2f} seconds\n")
        f.write(f"Best model: {best['model']}\n")


def main(debug: bool = False, clean_data: bool = True):
    t0 = time()

    print("===== PNEUMONIA DETECTION PIPELINE =====")
    print(f"Mode: {'DEBUG' if debug else 'PRODUCTION'}")
    print(f"Data cleaning: {'ENABLED' if clean_data else 'DISABLED'}")

    data_paths = run_data_cleaning(clean_data)
    results = run_training_pipeline(data_paths, debug)
    generate_final_report(results, time() - t0)

    print("\n===== PIPELINE COMPLETE =====")
    print(f"Total runtime: {time() - t0:.2f} seconds")
    print(f"Reports saved to: {MODELS_DIR / 'reports'}")
    print(f"Models saved to: {MODELS_DIR}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--skip-cleaning', action='store_true', help='Skip data cleaning phase')
    args = parser.parse_args()

    main(debug=args.debug, clean_data=not args.skip_cleaning)
