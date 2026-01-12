import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
from datetime import datetime
from config import Config
from src.model import build_training_model
from src.data_preprocessing import DataGenerator

def train_model():
    print("="*70)
    print("🇧🇩 Bangla Handwritten OCR - Training")
    print("="*70)
    print()
    
    # Check if data exists
    if not os.path.exists(Config.TRAIN_DIR):
        print(" Training data not found!")
        print()
        print("Please run: python download_dataset.py")
        return
    
    # Load data
    print(" Loading data...")
    train_gen = DataGenerator(Config.TRAIN_DIR, Config.BATCH_SIZE, augment=True)
    val_gen = DataGenerator(Config.VAL_DIR, Config.BATCH_SIZE, augment=False)
    
    print(f" Train samples: {len(train_gen.samples)}")
    print(f" Val samples: {len(val_gen.samples)}")
    print()
    
    if len(train_gen.samples) == 0:
        print(" No training samples found!")
        print("Check that data/train/annotations.json has entries")
        return
    
    # Build model
    print("  Building model...")
    training_model, pred_model = build_training_model()
    
    optimizer = tf.keras.optimizers.Adam(Config.LEARNING_RATE)
    training_model.compile(optimizer=optimizer)
    
    print()
    print("Model Summary:")
    print("-" * 70)
    pred_model.summary()
    print("-" * 70)
    print()
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"bangla_ocr_{timestamp}"
    
    callbacks = [
        ModelCheckpoint(
            os.path.join(Config.MODEL_DIR, f'{model_name}_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=Config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=Config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print(" Starting training...")
    print(f"   Epochs: {Config.EPOCHS}")
    print(f"   Batch size: {Config.BATCH_SIZE}")
    print(f"   Learning rate: {Config.LEARNING_RATE}")
    print()
    
    try:
        history = training_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=Config.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
    except KeyboardInterrupt:
        print("\n\n  Training interrupted by user")
        print("Partial model may be saved")
    except Exception as e:
        print(f"\n\n Training error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save final model
    final_path = os.path.join(Config.MODEL_DIR, f'{model_name}_final.h5')
    pred_model.save(final_path)
    
    print()
    print("="*70)
    print(" Training complete!")
    print("="*70)
    print()
    print(f" Model saved:")
    print(f"   Best: {Config.MODEL_DIR}/{model_name}_best.h5")
    print(f"   Final: {final_path}")
    print()
    print(" Next steps:")
    print("   - Test: python src/predict.py <model_path> <image_path>")
    print("   - Web App: python app/app.py")
    print()
    
    return history, pred_model

if __name__ == "__main__":
    train_model()