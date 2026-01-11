# Bangla Handwritten Paragraph Recognition

Deep learning model for recognizing handwritten Bangla text using CRNN architecture.

## Setup

1. Install dependencies:
````bash
pip install -r requirements.txt
````

2. Prepare dataset:
   - Place training images in `data/train/`
   - Place validation images in `data/val/`
   - Create `annotations.json` in each folder:
````json
[
  {"image": "img1.jpg", "text": "আপনার বাংলা টেক্সট"},
  {"image": "img2.jpg", "text": "আরো টেক্সট এখানে"}
]
````

## Training
````bash
python src/train.py
````

## Prediction
````bash
python src/predict.py models/saved_models/model.h5 path/to/image.jpg
````

## Web App
````bash
python app/app.py
````

Visit: http://localhost:5000

## Model Architecture

- CNN: Feature extraction from images
- RNN: Sequence modeling with Bidirectional LSTM
- CTC Loss: Alignment-free training

## 📊 Bangla Handwriting Datasets

### **Recommended Datasets:**

1. **BanglaWriting Dataset**
   - 🔗 [Kaggle - BanglaWriting](https://www.kaggle.com/datasets/nibaran/banglawriting)
   - 📝 47,000+ handwritten Bangla words
   - ✅ Good for word-level recognition

2. **BN-HTRd (Bangla Handwritten Text Recognition)**
   - 🔗 [GitHub - BN-HTRd](https://github.com/ai-ar/BN-HTRd)
   - 📝 Paragraph-level handwritten text
   - ✅ Perfect for paragraph recognition

3. **CMATERdb (Bangla Handwritten)**
   - 🔗 [CMATERdb 3.1.1](https://code.google.com/archive/p/cmaterdb/)
   - 📝 15,000+ character images
   - ✅ Good for character-level training

4. **Ekush (Bengali Handwritten Dataset)**
   - 🔗 [Ekush Dataset](https://www.kaggle.com/datasets/BengaliAI/ekush)
   - 📝 Large-scale handwritten characters
   - ✅ Bengali.AI competition dataset

5. **BanglaLekha-Isolated**
   - 🔗 [Kaggle - BanglaLekha](https://www.kaggle.com/datasets/mitulkumar/banglalekha-isolated)
   - 📝 166,000+ handwritten characters
   - ✅ Great for character recognition

### **How to Use:**

**Option 1: Download manually**
````bash
# After downloading, place in data/ folder
data/
├── train/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── annotations.json
└── val/
    ├── img001.jpg
    └── annotations.json
````

**Option 2: Use Kaggle API**
````bash
pip install kaggle

# Download BanglaWriting
kaggle datasets download -d nibaran/banglawriting

# Download Ekush
kaggle datasets download -d BengaliAI/ekush
````

### **Creating Your Own Dataset:**

1. Collect handwritten samples (phone camera works!)
2. Use tools like [Label Studio](https://labelstud.io/) or [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/)
3. Create `annotations.json` format:
````json
[
  {"image": "sample1.jpg", "text": "আমি বাংলা লিখি"},
  {"image": "sample2.jpg", "text": "এটি একটি উদাহরণ"}
]
````

## Tips

- Use at least 5000+ samples for good results
- Ensure consistent image quality
- Balance dataset across different writing styles
- Use data augmentation
- Mix datasets for better generalization