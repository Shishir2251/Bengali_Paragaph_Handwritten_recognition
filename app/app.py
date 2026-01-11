from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import BanglaOCRPredictor
from config import Config

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'app/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model (update path to your trained model)
MODEL_PATH = os.path.join(Config.MODEL_DIR, 'bangla_ocr_best.h5')
predictor = None

if os.path.exists(MODEL_PATH):
    predictor = BanglaOCRPredictor(MODEL_PATH)
    print("Model loaded successfully!")
else:
    print(f"Warning: Model not found at {MODEL_PATH}")
    print("Please train the model first or update MODEL_PATH")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Predict
            result = predictor.predict_image(filepath)
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'text': result
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': predictor is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)