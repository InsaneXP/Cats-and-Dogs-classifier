# 🐱🐶 Cat-Dog Classifier using ResNet & CNN

## 📌 Overview

This project is a part of my Major Project undertaken during my tenure as a **Project Trainee at CSIR-CMERI**. The primary objective is to develop an **image classification system** that distinguishes between cats and dogs using state-of-the-art **Deep Learning models**. In this initial phase, we focus on **Convolutional Neural Networks (CNN)** and **ResNet (Residual Networks)** for building an accurate classifier.

---

## 🚀 Tech Stack

- **Python 3.8+**
- **PyTorch / TensorFlow (any one)**
- **OpenCV**
- **NumPy**
- **Matplotlib / Seaborn**
- **Jupyter Notebook / Google Colab**

---

## 🧠 Model Architecture

### ✅ Phase 1: CNN-based Classifier
- Basic convolutional layers
- Max pooling
- Dropout layers for regularization
- Dense layers with softmax activation

### ✅ Phase 2: Transfer Learning using ResNet
- Pre-trained **ResNet-18 / ResNet-50**
- Modified classifier head for binary classification (cat/dog)
- Fine-tuning selected layers

---

## 🗂️ Dataset

- **Kaggle Cats and Dogs Dataset**: 25,000 labeled images
- Train/Test Split: 80/20
- Image Preprocessing:
  - Resizing to 224x224
  - Normalization
  - Data Augmentation (Flip, Rotation, Zoom)

---

## 📊 Performance Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 📷 Sample Outputs

| Input Image | Predicted Label |
|-------------|------------------|
| ![Cat](samples/cat1.jpg) | Cat |
| ![Dog](samples/dog1.jpg) | Dog |

---

## 🏗️ Project Structure

CatDogClassifier/
│
├── data/
│ ├── train/
│ └── test/
├── models/
│ └── resnet_model.pth
├── notebooks/
│ └── training_notebook.ipynb
├── utils/
│ └── data_loader.py
│ └── model_utils.py
├── samples/
│ └── cat1.jpg
│ └── dog1.jpg
├── README.md


##📍 Future Scope
- Expand to multi-class classification (different breeds)

- Deploy as a web app using Flask/Streamlit

- Integrate with real-time camera feed

👨‍💻 Author
Rohit Kumar Sahu
B.Tech in Information Technology
Project Trainee at CSIR-CMERI

📧 [Mail](r.sahu.2k17@gmail.com)
🔗 [Portfolio](https://rk-portfolio-pdio4kpi9-insanexps-projects.vercel.app/)

The portfolio is static in future I will try to add some interesting features! 



