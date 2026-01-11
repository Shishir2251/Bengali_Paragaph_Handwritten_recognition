import tensorflow as tf
import numpy as np
from config import Config
from src.utils import preprocess_image, decode_prediction

class BanglaOCRPredictor:
    def __init__(self, model_path):
        print(f"Loading: {model_path}")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded!")
    
    def predict_image(self, image_path):
        img = preprocess_image(image_path)
        if img is None:
            return "Error loading image"
        
        img = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img, verbose=0)
        
        input_len = np.ones(prediction.shape[0]) * prediction.shape[1]
        results = tf.keras.backend.ctc_decode(prediction, input_length=input_len, greedy=True)[0][0]
        
        return decode_prediction(results[0].numpy())

def test_prediction(model_path, image_path):
    predictor = BanglaOCRPredictor(model_path)
    result = predictor.predict_image(image_path)
    print(f"\nImage: {image_path}")
    print(f"Predicted: {result}")
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model_path> <image_path>")
        sys.exit(1)
    test_prediction(sys.argv[1], sys.argv[2])