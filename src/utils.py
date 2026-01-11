import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
import numpy as np
import cv2
from config import Config

def encode_text(text):
    """Convert text to numeric sequence"""
    encoded = []
    for char in text:
        if char in Config.CHAR_TO_IDX:
            encoded.append(Config.CHAR_TO_IDX[char])
    return encoded

def decode_prediction(prediction):
    """Convert numeric prediction to text"""
    text = ''
    for idx in prediction:
        if 0 <= idx < len(Config.IDX_TO_CHAR):
            text += Config.IDX_TO_CHAR[idx]
    return text

def preprocess_image(image_path):
    """Load and preprocess image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    img = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    
    return img

class CTCLayer(tf.keras.layers.Layer):
    """Custom CTC layer for training"""
    
    def __init__(self, name=None):
        super().__init__(name=name)
        
    def call(self, y_true, y_pred):
        # Get batch size
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int32")
        
        # Input length is the time dimension of predictions
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int32")
        input_length = input_length * tf.ones(shape=(batch_len,), dtype="int32")
        
        # Label length - count non-zero elements in each label
        label_length = tf.cast(tf.reduce_sum(tf.cast(y_true != 0, dtype="int32"), axis=1), dtype="int32")
        
        # Compute CTC loss
        loss = tf.nn.ctc_loss(
            labels=tf.cast(y_true, dtype="int32"),
            logits=y_pred,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,
            blank_index=-1
        )
        
        # Add loss to layer
        self.add_loss(tf.reduce_mean(loss))
        
        # Return predictions unchanged
        return y_pred