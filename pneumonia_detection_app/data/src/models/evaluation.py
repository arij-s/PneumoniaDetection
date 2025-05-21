from typing import List
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config.paths import MODELS_DIR
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class ModelEvaluator:
    @staticmethod
    def evaluate(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Thread-safe model evaluation with comprehensive metrics
        
        Args:
            name: Model name
            model: Trained model instance
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Initialize metrics with default values
        metrics = {
            'model': name,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'auc': 0,
            'average_precision': 0
        }
        
        try:
            # Basic predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics with zero_division handling
            metrics.update({
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
            })
            
            # Probability metrics if available
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    metrics.update({
                        'auc': roc_auc_score(y_test, y_proba),
                        'average_precision': average_precision_score(y_test, y_proba)
                    })
                    ModelEvaluator._save_roc_curve(name, y_test, y_proba)
                    ModelEvaluator._save_pr_curve(name, y_test, y_proba)
                except Exception as e:
                    print(f"Couldn't calculate probability metrics for {name}: {str(e)}")
            
            # Generate and save plots
            ModelEvaluator._save_confusion_matrix(name, y_test, y_pred)
            
            # Save classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            with open(MODELS_DIR / f"{name}_classification_report.json", 'w') as f:
                json.dump(report, f, indent=2)
            
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
        
        print(f"\n=== {name} Evaluation ===")
        print(pd.DataFrame([metrics]).T.to_string(header=False))
        
        return metrics

    @staticmethod
    def _save_confusion_matrix(name: str, y_true: np.ndarray, y_pred: np.ndarray):
        """Save normalized confusion matrix"""
        try:
            plt.figure(figsize=(6, 5))
            cm = confusion_matrix(y_true, y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=['Normal', 'Pneumonia'],
                        yticklabels=['Normal', 'Pneumonia'])
            plt.title(f"{name} Normalized Confusion Matrix")
            plt.tight_layout()
            plt.savefig(MODELS_DIR / f"{name}_confusion_matrix.png")
            plt.close()
        except Exception as e:
            print(f"Couldn't save confusion matrix for {name}: {str(e)}")

    @staticmethod
    def _save_roc_curve(name: str, y_true: np.ndarray, y_proba: np.ndarray):
        """Save ROC curve plot"""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = roc_auc_score(y_true, y_proba)
            
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{name} Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(MODELS_DIR / f"{name}_roc_curve.png")
            plt.close()
        except Exception as e:
            print(f"Couldn't save ROC curve for {name}: {str(e)}")

    @staticmethod
    def _save_pr_curve(name: str, y_true: np.ndarray, y_proba: np.ndarray):
        """Save Precision-Recall curve"""
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            avg_precision = average_precision_score(y_true, y_proba)
            
            plt.figure()
            plt.plot(recall, precision, color='blue', lw=2,
                     label=f'AP={avg_precision:.2f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title(f'{name} Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.savefig(MODELS_DIR / f"{name}_pr_curve.png")
            plt.close()
        except Exception as e:
            print(f"Couldn't save PR curve for {name}: {str(e)}")

   

    @staticmethod
    def save_results(results: list):
        """Save evaluation results to CSV and create visual comparison"""
        try:
            df = pd.DataFrame(results)
            df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
            
            # Create a visual comparison
            plt.figure(figsize=(12, 6))
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
            
            for i, metric in enumerate(metrics):
                plt.subplot(1, len(metrics), i+1)
                plt.bar(df['model'], df[metric])
                plt.title(metric.replace('_', ' ').title())
                plt.xticks(rotation=45)
                plt.ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(MODELS_DIR / "model_comparison.png")
            plt.close()
        except Exception as e:
            print(f"Couldn't save results: {str(e)}")
    



