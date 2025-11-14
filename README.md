# Smart-pothole-detection-system
  A deep learning–based solution for automatic pothole detection using Convolutional Neural Networks (CNNs) and image processing. This project leverages Python, OpenCV, and TensorFlow/Keras to detect potholes from road images or video frames with high accuracy. The system preprocesses input images, extracts features using a trained CNN model.  
### by **Simerjeet Tech**

The main approach is:

## ⭐ 1. CNN Classifier + OpenCV (Primary Method)
- OpenCV finds regions that *look like potholes*
- CNN (MobileNetV2) classifies each region as pothole / not pothole

This repository also includes:

## ⭐ 2. YOLOv8 Object Detection (Advanced Module)
- Detects potholes with bounding boxes (bonus module)

## ⭐ 3. Streamlit Web App
- Upload an image → get pothole detection instantly

## ⭐ 4. Google Colab Notebooks (Modular)
- Training Notebook  
- Evaluation Notebook  
- Inference Notebook  
- YOLOv8 Notebook  

---

# 📁 Project Structure

```
smart-pothole-detection/
│
├── src/                         # Main detection system
│   ├── data_prep.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference_cv_and_cnn.py
│   └── visualize.py
│
├── object_detection_yolo/       # YOLOv8 bonus module
│   ├── yolo_train.py
│   └── yolo_inference.py
│
├── webapp/                      # Streamlit App
│   ├── app.py
│   └── utils.py
│
├── notebooks/                   # All Colab notebooks
│   ├── CNN Training (link)
│   ├── Evaluation (link)
│   ├── Inference (link)
│   └── YOLO Training (link)
│
├── examples/                    # Example images (add your own)
│
├── requirements.txt
├── Dockerfile
├── LICENSE
└── README.md
```

---

# 🚀 Getting Started

## 1. Install dependencies
```
pip install -r requirements.txt
```

## 2. Train CNN Model
```
python src/train.py --data_dir data --epochs 20
```

## 3. Evaluate
```
python src/evaluate.py --model_path models/best_model.h5
```

## 4. Run Inference (CNN + OpenCV)
```
python src/inference_cv_and_cnn.py --model_path models/best_model.h5 --input examples/
```

---

# 🌐 Run the Web App

```
streamlit run webapp/app.py
```

---

# 🎯 YOLOv8 Training (Advanced)

```
python object_detection_yolo/yolo_train.py
```

---

# ✨ Author  
**Simerjeet Tech**

---

# 📝 License  
MIT License
