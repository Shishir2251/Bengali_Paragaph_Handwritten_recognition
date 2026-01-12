"""
Auto-label using EasyOCR (Easiest!)
Requires: pip install easyocr
Works out of the box, no API keys needed
"""

import json
from pathlib import Path
import easyocr
from tqdm import tqdm

def auto_label_with_easyocr():
    """Auto-label using EasyOCR"""
    
    print("Initializing EasyOCR (first time takes a while)...")
    reader = easyocr.Reader(['bn'], gpu=False)  # 'bn' = Bengali
    
    data_dir = Path('data')
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        ann_file = split_dir / 'annotations.json'
        
        if not ann_file.exists():
            continue
        
        print(f"\n📂 Processing {split}...")
        
        with open(ann_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        for item in tqdm(annotations):
            img_path = str(split_dir / item['image'])
            
            try:
                # OCR
                results = reader.readtext(img_path)
                
                # Combine all detected text
                text = ' '.join([result[1] for result in results])
                
                if text.strip():
                    item['text'] = text.strip()
                else:
                    item['text'] = "NO_TEXT_DETECTED"
                    
            except Exception as e:
                print(f"Error with {img_path}: {e}")
                item['text'] = "ERROR"
        
        # Save
        with open(ann_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Labeled {len(annotations)} images")

if __name__ == "__main__":
    print("="*70)
    print("🤖 Auto-labeling with EasyOCR")
    print("="*70)
    print()
    
    auto_label_with_easyocr()
    
    print("\n✅ Auto-labeling complete!")
    print("\nNext: python src/train.py")