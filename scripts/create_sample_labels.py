"""
Create sample labels for testing
WARNING: This creates dummy labels - replace with real ones!
"""

import json
from pathlib import Path

sample_texts = [
    "আমি বাংলায় গান গাই",
    "বাংলা আমার মাতৃভাষা",
    "সে স্কুলে যায়",
    "আমরা বই পড়ি",
    "তুমি কোথায় যাও",
]

for split in ['train', 'val', 'test']:
    split_dir = Path('data') / split
    
    if not split_dir.exists():
        continue
    
    # Get all images
    images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
    
    annotations = []
    for idx, img_path in enumerate(images):
        # Use sample text (cycling through)
        text = sample_texts[idx % len(sample_texts)]
        
        annotations.append({
            "image": img_path.name,
            "text": text
        })
    
    # Save
    ann_file = split_dir / "annotations.json"
    with open(ann_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Created {split}/annotations.json with {len(annotations)} entries")

print()
print("⚠️  WARNING: These are DUMMY labels for testing!")
print("Replace with real labels for actual training.")