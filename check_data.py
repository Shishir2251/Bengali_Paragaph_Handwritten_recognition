"""
Quick script to check your data folder structure
"""

import os
from pathlib import Path
import json

def check_data_structure():
    print("="*70)
    print(" Checking Your Data Folder")
    print("="*70)
    print()
    
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("'data' folder not found!")
        return
    
    print(f"Found data folder: {data_dir.absolute()}")
    print()
    
    # Check each split
    for split in ['train', 'val', 'test']:
        split_path = data_dir / split
        
        if split_path.exists():
            print(f" {split}/")
            
            # Count images
            images = list(split_path.glob("*.jpg")) + list(split_path.glob("*.png"))
            print(f"   Images: {len(images)}")
            
            # Check for annotations.json
            ann_file = split_path / "annotations.json"
            if ann_file.exists():
                with open(ann_file, 'r', encoding='utf-8') as f:
                    anns = json.load(f)
                print(f" annotations.json: {len(anns)} entries")
            else:
                print(f"  annotations.json NOT FOUND")
            
            print()
        else:
            print(f" {split}/ folder not found")
            print()
    
    print("="*70)
    
    # Check if ready for training
    ready = True
    for split in ['train', 'val']:
        ann_file = data_dir / split / "annotations.json"
        if not ann_file.exists():
            ready = False
    
    if ready:
        print(" Your dataset is READY for training!")
        print()
        print(" Next step:")
        print("   python src/train.py")
    else:
        print("  You need to create annotations.json files")
        print()
        print(" Choose one option:")
        print()
        print("Option A: If you have labels in separate JSON files:")
        print("   1. Put them in data/train/labels/, data/val/labels/")
        print("   2. Run: python scripts/build_annotations.py")
        print()
        print("Option B: Create annotations.json manually:")
        print('   Format: [{"image": "img1.jpg", "text": "বাংলা টেক্সট"}]')
        print()
        print("Option C: If images have text in filenames:")
        print("   Run: python scripts/create_annotations_from_filenames.py")
    
    print()

if __name__ == "__main__":
    check_data_structure()