from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier,
    AdaBoostClassifier,
)

from sklearn.neural_network import MLPClassifier


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from joblib import dump
from config.paths import MODELS_DIR
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class ModelTrainer:
    def __init__(self):
        """Initialize models with optimized hyperparameter grids"""
        self.models = {}

    def _init_models(self, y_train: np.ndarray):
        """Initialize models with class weights based on training data"""
        # Calculate class weights if data is imbalanced
        class_ratio = len(y_train[y_train==0])/len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1
        
        self.models = {
            'RandomForest': (
                RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1),
                {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'max_features': ['sqrt', 'log2'],
                    'class_weight': ['balanced', None]
                }
            ),
            
            'MLP': (
                Pipeline([
                    ('scaler', StandardScaler()),
                    ('mlp', MLPClassifier(random_state=42, early_stopping=True))
                ]),
                {
                    'mlp__hidden_layer_sizes': [(100,), (100,50)],
                    'mlp__alpha': [0.0001, 0.001, 0.01],
                    'mlp__learning_rate': ['constant', 'adaptive']
                }
            ),
            'AdaBoost': (
                AdaBoostClassifier(random_state=42),
                {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                }
            ),
            'ExtraTrees': (
                ExtraTreesClassifier(class_weight='balanced', random_state=42, n_jobs=1),
                {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
            )
        }

    def safe_tune_and_train(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray, 
                          n_iter: int = 10) -> dict:
        """
        Thread-safe model training with leakage prevention
        """
        # Validate class distribution
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            raise ValueError(f"Training data contains only one class: {unique_classes}. "
                          f"Need at least 2 classes for classification.")
        
        print("\nClass distribution in training set:")
        for cls in [0, 1]:
            print(f"Class {cls}: {sum(y_train == cls)} samples ({sum(y_train == cls)/len(y_train):.1%})")
        
        # Initialize models with proper class weights
        self._init_models(y_train)
        
        # Create validation split
        val_fold = np.zeros(len(X_train))
        val_fold[:len(X_train)//3] = -1  # 33% validation set
        ps = PredefinedSplit(val_fold)
        
        trained_models = {}
        
        for name, (model, params) in self.models.items():
            print(f"\n=== Training {name} ===")
            
            try:
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=params,
                    n_iter=n_iter,
                    cv=ps,
                    scoring='roc_auc',
                    n_jobs=1,
                    verbose=1,
                    random_state=42
                )
                
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                
                # Save model
                model_path = MODELS_DIR / f"{name}.joblib"
                dump(best_model, model_path)
                print(f"Best params: {search.best_params_}")
                print(f"Saved to {model_path}")
                
                trained_models[name] = best_model
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
                
        return trained_models

    def tune_and_train(self, X_train: np.ndarray, y_train: np.ndarray, 
                      cv: int = 3, n_iter: int = 10) -> dict:
        """
        Original training method (not thread-safe)
        """
        warnings.warn("This method may cause threading issues. Use safe_tune_and_train instead.")
        return self.safe_tune_and_train(X_train, y_train, None, None, n_iter)