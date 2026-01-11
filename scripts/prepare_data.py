"""
Prepare your custom dataset for training
Copies images and uses existing annotations
"""

import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

class CustomDatasetPreparator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.raw_data = self.project_root / "raw_data" / "converted"
        self.data_dir = self.project_root / "data"
    
    # ... (rest of code from artifact)