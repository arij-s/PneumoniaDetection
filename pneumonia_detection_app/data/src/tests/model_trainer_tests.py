# model_trainer_tests.py
import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

# Import the class to test
from models.training import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.trainer = ModelTrainer()
        
        # Create dummy data
        self.X_train = np.random.rand(100, 50)  # 100 samples, 50 features
        self.y_train = np.zeros(100)
        self.y_train[:50] = 1  # 50 samples of class 0, 50 samples of class 1
        
        self.X_test = np.random.rand(20, 50)
        self.y_test = np.zeros(20)
        self.y_test[:10] = 1  # 10 samples of class 0, 10 samples of class 1
        
        # Create a temporary models directory
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_init_models(self):
        """Test initialization of models"""
        self.trainer._init_models(self.y_train)
        
        # Check that all models were initialized
        self.assertIn('RandomForest', self.trainer.models)
        self.assertIn('MLP', self.trainer.models)
        self.assertIn('AdaBoost', self.trainer.models)
        self.assertIn('ExtraTrees', self.trainer.models)
        
        # Check model parameters
        rf_model, rf_params = self.trainer.models['RandomForest']
        self.assertEqual(rf_model.random_state, 42)
        self.assertEqual(rf_model.class_weight, 'balanced')
        
        # Check param grids
        self.assertIn('n_estimators', rf_params)
        self.assertIn('max_depth', rf_params)
        
    def test_init_models_imbalanced(self):
        """Test initialization of models with imbalanced classes"""
        # Create imbalanced dataset
        y_imbalanced = np.zeros(100)
        y_imbalanced[:10] = 1  # 10% positive samples
        
        self.trainer._init_models(y_imbalanced)
        
        # Make sure models were initialized
        self.assertGreater(len(self.trainer.models), 0)
        
    @patch('models.training.RandomizedSearchCV')
    @patch('models.training.dump')
    @patch('config.paths.MODELS_DIR', None)  # Will be replaced in the test
    def test_safe_tune_and_train(self, mock_dump, mock_cv):
        """Test safe tuning and training process"""
        # Set up mocks
        mock_best_estimator = MagicMock()
        mock_cv_instance = MagicMock()
        mock_cv.return_value = mock_cv_instance
        mock_cv_instance.best_estimator_ = mock_best_estimator
        mock_cv_instance.best_params_ = {'param1': 'value1'}
        
        # Replace MODELS_DIR with our temp directory
        with patch('config.paths.MODELS_DIR', self.temp_dir):
            # Call the method
            trained_models = self.trainer.safe_tune_and_train(
                self.X_train, self.y_train, self.X_test, self.y_test, n_iter=5
            )
            
            # Check that RandomizedSearchCV was called for each model
            self.assertEqual(mock_cv.call_count, 4)  # One for each model type
            
            # Check that models were saved
            self.assertEqual(mock_dump.call_count, 4)  # One for each model type
            
            # Check that trained models were returned
            self.assertEqual(len(trained_models), 4)
    
    def test_safe_tune_and_train_single_class(self):
        """Test exception is raised when training data has only one class"""
        # Create data with only one class
        y_single_class = np.zeros(100)
        
        with self.assertRaises(ValueError):
            self.trainer.safe_tune_and_train(
                self.X_train, y_single_class, self.X_test, self.y_test
            )
    
    @patch('models.training.RandomizedSearchCV')
    def test_safe_tune_and_train_exception_handling(self, mock_cv):
        """Test that exceptions in model training are properly handled"""
        # Configure the mock to raise an exception
        mock_cv_instance = MagicMock()
        mock_cv.return_value = mock_cv_instance
        mock_cv_instance.fit.side_effect = Exception("Training error")
        
        # Replace MODELS_DIR with our temp directory
        with patch('config.paths.MODELS_DIR', self.temp_dir):
            # Call the method - it should continue despite exceptions
            trained_models = self.trainer.safe_tune_and_train(
                self.X_train, self.y_train, self.X_test, self.y_test, n_iter=5
            )
            
            # No models should have been trained successfully
            self.assertEqual(len(trained_models), 0)
    
    @patch('models.training.warnings.warn')
    @patch('models.training.ModelTrainer.safe_tune_and_train')
    def test_tune_and_train_warning(self, mock_safe_method, mock_warn):
        """Test that the deprecated method issues a warning"""
        # Call the deprecated method
        self.trainer.tune_and_train(self.X_train, self.y_train)
        
        # Check that a warning was issued
        mock_warn.assert_called_once()
        self.assertIn("threading issues", mock_warn.call_args[0][0])
        
        # Check that the safe method was called
        mock_safe_method.assert_called_once()
    
    @patch('models.training.PredefinedSplit')
    def test_validation_split(self, mock_split):
        """Test that validation split is created correctly"""
        with patch('models.training.RandomizedSearchCV'):
            with patch('config.paths.MODELS_DIR', self.temp_dir):
                with patch('models.training.dump'):
                    self.trainer.safe_tune_and_train(
                        self.X_train, self.y_train, self.X_test, self.y_test
                    )
                    
                    # Check that PredefinedSplit was called
                    mock_split.assert_called_once()
                    
                    # Get the validation fold array
                    val_fold = mock_split.call_args[0][0]
                    
                    # Check that about 1/3 of the data is used for validation
                    expected_val_samples = len(self.X_train) // 3
                    actual_val_samples = np.sum(val_fold == -1)
                    self.assertEqual(actual_val_samples, expected_val_samples)