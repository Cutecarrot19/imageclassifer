# 🧠 Custom Image Classifier using CNN

A beginner-friendly machine learning project that classifies images into custom categories using a Convolutional Neural Network (CNN). Built with TensorFlow and Python.

## 🚀 Features

- Train your own image classifier with 10–15 images per class
- Recognizes test images and predicts the correct class
- Easily extendable to new classes
- Clean, beginner-friendly code
- Optional: Upgradeable to web app using Flask

---

## 🗂️ Folder Structure

```
image_classifer/
├── dataset/
│   ├── class1/         # e.g., cats
│   ├── class2/         # e.g., dogs
├── train_model.py      # Train CNN on dataset
├── predict.py          # Test with new images
├── test.jpg            # Test image
├── image_classifier_model.h5  # Trained model (auto-generated)
├── README.md           # Project readme
├── requirements.txt    # Required Python packages
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/image-classifier.git
cd image-classifier
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

- Place your images inside the `dataset/` folder.
- One folder per class (e.g., `cats/`, `dogs/`)
- Minimum 10–15 images per class recommended

### 4. Train the Model

```bash
python train_model.py
```

- Saves the model as `image_classifier_model.h5`

### 5. Test with New Image

- Put your test image in the root folder as `test.jpg`

```bash
python predict.py
```

- Outputs the predicted class

---

## 📦 requirements.txt

```txt
tensorflow
numpy
pillow
```

---

## 📈 Future Improvements

- Add Flask-based drag-and-drop web interface
- Improve accuracy with Transfer Learning (e.g., MobileNetV2)
- Deploy model as web API or Android app (TFLite)

---

## 🤝 Credits

Built with 💻 Python and 🧠 TensorFlow.  
Created as a personal ML project to learn computer vision.

---


