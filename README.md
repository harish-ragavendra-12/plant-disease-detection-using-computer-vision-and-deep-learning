# 🌿 Plant Disease Detection Using Computer Vision & Deep Learning

## 📌 Project Overview

Plant diseases significantly affect agricultural productivity and crop quality. Early identification of plant diseases helps farmers take preventive measures and reduce crop losses.

This project presents an intelligent **Plant Disease Detection System** built using **Computer Vision** and **Deep Learning**. The system utilizes **Transfer Learning with MobileNetV2** to classify plant leaf images into different disease categories. A professional **Streamlit web application** allows users to upload a plant leaf image and receive instant disease predictions with confidence scores.

---

## 🎯 Problem Statement

Manual identification of plant diseases is time-consuming and requires expert knowledge. Farmers often struggle to identify diseases during the early stages, leading to delayed treatment and reduced crop yield.

The objective of this project is to build an AI-powered application capable of automatically detecting plant diseases from leaf images with high accuracy.

---

## 🚀 Features

- Deep Learning based plant disease detection
- Transfer Learning using MobileNetV2
- Image preprocessing and normalization
- Dataset visualization (EDA)
- Model evaluation using multiple metrics
- Confusion Matrix
- Classification Report
- Accuracy & Loss Curves
- Single image prediction
- Confidence score for predictions
- Top-3 predicted diseases
- Disease description
- Treatment recommendations
- Prevention tips
- Professional Streamlit Dashboard

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Keras
- MobileNetV2

### Computer Vision

- OpenCV
- Pillow

### Data Analysis

- NumPy
- Pandas

### Visualization

- Matplotlib
- Seaborn

### Machine Learning Utilities

- Scikit-learn

### Web Application

- Streamlit

### Version Control

- Git
- GitHub

---

## 📂 Project Structure

```
plant-disease-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── figures/
│
├── models/
│
├── src/
│   ├── config.py
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── prediction.py
│   ├── disease_info.py
│   └── utils.py
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

**Dataset Name**

PlantVillage Dataset

The dataset contains thousands of labeled plant leaf images from different crop species including:

- Tomato
- Potato
- Pepper

Disease categories include:

- Healthy
- Early Blight
- Late Blight
- Leaf Mold
- Bacterial Spot
- Septoria Leaf Spot
- Target Spot
- Yellow Leaf Curl Virus
- Tomato Mosaic Virus
- Spider Mites

---

## 🔄 Project Workflow

```
Plant Leaf Images
        │
        ▼
Load Dataset
        │
        ▼
Image Preprocessing
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Transfer Learning
(MobileNetV2)
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Prediction
        │
        ▼
Streamlit Deployment
```

---

## 🧠 Model Architecture

- MobileNetV2 (Pre-trained on ImageNet)
- Global Average Pooling Layer
- Dropout Layer
- Dense Softmax Output Layer

---

## 📈 Model Evaluation

The model is evaluated using:

- Validation Accuracy
- Validation Loss
- Classification Report
- Confusion Matrix
- Accuracy Curve
- Loss Curve

---

## 🌱 Disease Prediction

The application predicts:

- Plant Disease
- Prediction Confidence
- Top-3 Predictions
- Disease Description
- Symptoms
- Treatment Recommendations
- Prevention Measures

---

## 💻 Streamlit Application

The web application allows users to:

- Upload a plant leaf image
- Preview the uploaded image
- Detect plant disease
- View prediction confidence
- Display Top-3 predictions
- Read disease information
- Learn treatment methods
- View prevention measures

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/harish-ragavendra-12/plant-disease-detection-using-computer-vision-and-deep-learning
```

Navigate to the project folder

```bash
cd plant-disease-detection-using-computer-vision-and-deep-learning
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run streamlit_app.py
```

---

## 📷 Sample Output

The application displays:

- Uploaded Plant Leaf
- Predicted Disease
- Confidence Score
- Top-3 Predictions
- Disease Information
- Treatment Suggestions
- Prevention Tips

---

## 📌 Future Enhancements

- Support additional crop species
- Real-time disease detection using webcam
- Mobile application deployment
- Cloud deployment using Streamlit Community Cloud
- Fine-tuning MobileNetV2 for improved accuracy
- Multi-language support
- Disease severity estimation
- Fertilizer recommendation system

---

## 📚 Learning Outcomes

This project helped in understanding:

- Computer Vision fundamentals
- Deep Learning using TensorFlow
- Transfer Learning
- MobileNetV2 Architecture
- Image Classification
- Data Preprocessing
- Model Evaluation
- Streamlit Deployment
- Git & GitHub Version Control

---

## 👨‍💻 Author

**Harish Ragavendra**

Aspiring Data Scientist | Machine Learning & Deep Learning Enthusiast

---

## ⭐ If you found this project useful, please consider giving it a star!