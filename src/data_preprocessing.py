import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import numpy as np
import tensorflow as tf
import json
import albumentations as A
from config import Config
from src.utils import encode_text, preprocess_image

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=16, augment=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.augment = augment
        self.samples = self.load_annotations()
        self.indexes = np.arange(len(self.samples))
        
        if self.augment:
            self.augmentor = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.Rotate(limit=5, p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            ])
    
    def load_annotations(self):
        """Load annotations.json created by build_annotations.py"""
        annotation_file = os.path.join(self.data_dir, 'annotations.json')
        
        if not os.path.exists(annotation_file):
            print(f" Error: {annotation_file} not found!")
            return []
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))
    
    def __getitem__(self, index):
        """Return batch as dictionary for training model with 2 inputs"""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_samples = [self.samples[i] for i in indexes]
        
        images, labels = self.generate_batch(batch_samples)
        
        # Return as dictionary with both inputs for the training model
        return {"image": images, "label": labels}, np.zeros((len(images),))
    
    def generate_batch(self, batch_samples):
        images, labels = [], []
        
        for sample in batch_samples:
            img_path = os.path.join(self.data_dir, sample['image'])
            text = sample['text']
            
            img = preprocess_image(img_path)
            if img is None:
                continue
            
            if self.augment:
                img = self.augmentor(image=(img*255).astype(np.uint8))['image']
                img = img.astype(np.float32) / 255.0
            
            encoded = encode_text(text)
            if len(encoded) > Config.MAX_TEXT_LENGTH:
                encoded = encoded[:Config.MAX_TEXT_LENGTH]
            
            padded = np.pad(encoded, (0, Config.MAX_TEXT_LENGTH - len(encoded)), constant_values=0)
            
            images.append(img)
            labels.append(padded)
        
        return np.array(images), np.array(labels)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)