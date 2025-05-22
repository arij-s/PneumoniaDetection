import json
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from features.extraction import FeatureExtractor

app = Flask(__name__,static_folder='../../static',static_url_path='/')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MODEL_DIR'] = 'pneumonia_detection_app/models'

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """Preprocess image to match training pipeline"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32) / 255.0
    return image

def get_best_model():
    """Find the best available model with proper fallbacks"""
    models = []
    model_dir = Path(app.config['MODEL_DIR'])
    
    # Check if model directory exists
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory {model_dir} does not exist")
    
    # Find all available model files
    model_files = list(model_dir.glob('*.joblib')) + list(model_dir.glob('*.pkl'))
    
    if not model_files:
        raise ValueError("No model files found (.joblib or .pkl) in models directory")
    
    for model_file in model_files:
        model_name = model_file.stem
        if '_metrics' in model_name:
            continue
            
        # Default metrics
        metrics = {
            'recall': 0.85,  # Conservative default
            'model_name': model_name
        }
        
        # Try to load metrics if available
        metrics_file = model_dir / f"{model_name}_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics.update(json.load(f))
            except json.JSONDecodeError:
                app.logger.warning(f"Invalid metrics file: {metrics_file}")
        
        models.append({
            'path': model_file,
            **metrics
        })
    
    if not models:
        raise ValueError("No valid models found after processing")
    
    # Select model with highest recall (or your preferred metric)
    best_model = max(models, key=lambda x: x['recall'])
    app.logger.info(f"Selected model: {best_model['model_name']} (Recall: {best_model['recall']})")
    
    return joblib.load(best_model['path']), best_model['model_name'], best_model['recall']

@app.route('/')
def home():
    return send_from_directory(app.static_folder,'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Preprocess image
    image = preprocess_image(filepath)
    if image is None:
        return jsonify({'error': 'Could not process image'}), 400
    
    try:
        # Load best model
        model, model_name, recall = get_best_model()
        
        # Extract features (simplified example)
          # Import your feature extractor
        feature_extractor = FeatureExtractor()
        features = feature_extractor.extract(image).reshape(1, -1)
        
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1]) if hasattr(model, 'predict_proba') else float(prediction)
        
        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'label': 'Pneumonia' if prediction == 1 else 'Normal',
            'image_path': f"/uploads/{filename}",
            'model_name': model_name,
            'recall': recall
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_DIR'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)