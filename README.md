# Silver-XRay14

**Note**: The dataset and model weights are too large to be uploaded to this repository. To run the YOLOv5 model , you must clone the original [YOLOv5 repository](https://github.com/ultralytics/yolov5) and add the YAML file from this repository. Afterwards you can run it using the RunYolo.py

Dataset used: https://www.kaggle.com/datasets/nih-chest-xrays/data

---

## **Project Structure**

### **Classification Folder**
This folder contains the following models for classification tasks:
- **CheXNet Code**
- **DenseNet Code**
- **EfficientNet Code**
- **ResNet Code**

Additionally, it includes:
- **Ensemble Code**: To perform both average and weighted ensemble methods across the classification models.

---

### **Object Detection Folder**
This folder includes models for object detection tasks:
- **Faster R-CNN Code**
- **YOLOv5 Code**: Runfile (config) + YAML file.

---

### **Components Folder**
Contains:
- **Dataset Class for Classification Models**: For handling and preparing data.
---
