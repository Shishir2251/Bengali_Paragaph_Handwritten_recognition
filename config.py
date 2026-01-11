import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'val')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
    
    # Model hyperparameters
    IMG_HEIGHT = 64
    IMG_WIDTH = 512
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 0.0001
    MAX_TEXT_LENGTH = 128
    
    # Bangla characters
    CHARACTERS = (
        'অ আ ই ঈ উ ঊ ঋ এ ঐ ও ঔ '
        'ক খ গ ঘ ঙ চ ছ জ ঝ ঞ ট ঠ ড ঢ ণ ত থ দ ধ ন প ফ ব ভ ম য র ল শ ষ স হ '
        'ড় ঢ় য় ৎ ং ঃ ঁ া ি ী ু ূ ৃ ে ৈ ো ৌ ্ '
        '০ ১ ২ ৩ ৪ ৫ ৬ ৭ ৮ ৯ '
        ' '
    )
    
    CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARACTERS)}
    IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARACTERS)}
    NUM_CLASSES = len(CHARACTERS) + 1
    
    # Training
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 7
    USE_AUGMENTATION = True

os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.TRAIN_DIR, exist_ok=True)
os.makedirs(Config.VAL_DIR, exist_ok=True)
os.makedirs(Config.TEST_DIR, exist_ok=True)