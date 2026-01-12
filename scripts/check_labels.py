"""
Check auto-labeled data quality
"""

import json
from pathlib import Path

def check_labels():
    print("="*70)
    print("Checking Auto-Labeled Data")
    print("="*70)
    
    for split in ['train', 'val', 'test']:
        ann_file = Path('data') / split / 'annotations.json'
        
        if not ann_file.exists():
            continue
        
        with open(ann_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        print(f"\n📂 {split.upper()}:")
        print(f"   Total samples: {len(annotations)}")
        
        # Count issues
        no_text = sum(1 for a in annotations if a['text'] in ['NO_TEXT_DETECTED', 'ERROR', '', 'LABEL_NEEDED'])
        very_short = sum(1 for a in annotations if len(a['text']) < 3)
        avg_len = sum(len(a['text']) for a in annotations) / len(annotations) if annotations else 0
        
        print(f"   No text detected: {no_text} ({no_text/len(annotations)*100:.1f}%)")
        print(f"   Very short (<3 chars): {very_short} ({very_short/len(annotations)*100:.1f}%)")
        print(f"   Average length: {avg_len:.1f} characters")
        
        print(f"\n   First 5 samples:")
        for i, item in enumerate(annotations[:5]):
            text_preview = item['text'][:60] + '...' if len(item['text']) > 60 else item['text']
            print(f"     {i+1}. {item['image']}")
            print(f"        Text: '{text_preview}'")
            print(f"        Length: {len(item['text'])} chars")
    
    print("\n" + "="*70)
    print("Analysis:")
    print("="*70)
    
    # Load train data for detailed analysis
    with open('data/train/annotations.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # Check character distribution
    all_text = ''.join([a['text'] for a in train_data])
    bangla_chars = sum(1 for c in all_text if '\u0980' <= c <= '\u09FF')
    english_chars = sum(1 for c in all_text if c.isalpha() and c.isascii())
    
    print(f"\nCharacter Analysis:")
    print(f"  Total characters: {len(all_text)}")
    print(f"  Bangla characters: {bangla_chars} ({bangla_chars/len(all_text)*100:.1f}%)" if all_text else "  No text")
    print(f"  English characters: {english_chars} ({english_chars/len(all_text)*100:.1f}%)" if all_text else "  No text")
    
    # Show unique first characters (to detect language)
    first_chars = set([a['text'][0] for a in train_data if a['text']])
    print(f"\nFirst characters in labels: {list(first_chars)[:20]}")

if __name__ == "__main__":
    check_labels()