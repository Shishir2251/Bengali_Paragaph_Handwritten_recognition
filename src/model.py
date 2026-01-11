import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
from tensorflow.keras import layers, Model
from config import Config

def build_crnn_model():
    input_img = layers.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 1), name='image')
    
    # CNN Feature Extraction
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)  # 64 -> 32, 512 -> 256
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 32 -> 16, 256 -> 128
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 16 -> 8, 128 -> 64
    
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)  # 8 -> 4, 64 -> 64
    
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    
    # Calculate shape after convolutions
    # Input: 64 x 512 x 1
    # After pooling: 4 x 64 x 512
    # Reshape to: (64, 2048) where 2048 = 4 * 512
    
    x = layers.Reshape(target_shape=(64, 2048))(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(x)
    
    # Output layer
    output = layers.Dense(Config.NUM_CLASSES, activation='softmax')(x)
    
    return Model(inputs=input_img, outputs=output, name='CRNN_Bangla_OCR')

def build_training_model():
    from src.utils import CTCLayer
    
    base_model = build_crnn_model()
    labels = layers.Input(name='label', shape=(Config.MAX_TEXT_LENGTH,), dtype='float32')
    output = CTCLayer(name='ctc_loss')(labels, base_model.output)
    
    return Model(inputs=[base_model.input, labels], outputs=output), base_model