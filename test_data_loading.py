"""
Test if data loading works
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing data loading...")
print()

from config import Config
from src.data_preprocessing import DataGenerator

print("1. Loading train generator...")
train_gen = DataGenerator(Config.TRAIN_DIR, batch_size=4, augment=False)
print(f"   ✓ Train samples: {len(train_gen.samples)}")

print("\n2. Getting first batch...")
try:
    X, y = train_gen[0]
    print(f"   ✓ Batch shape: X={X.shape}, y={y.shape}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Loading val generator...")
val_gen = DataGenerator(Config.VAL_DIR, batch_size=4, augment=False)
print(f"   ✓ Val samples: {len(val_gen.samples)}")

print("\n✅ Data loading works!")