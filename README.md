# Silver-XRay14

**Note**: The dataset and model weights are too large to be uploaded to this repository. Only the specific YOLOv5 configuration files have been included. To run YOLOv5 with these configurations, you must clone the original [YOLOv5 repository](https://github.com/ultralytics/yolov5) and add the provided configuration files from this project.

Dataset used: https://www.kaggle.com/datasets/nih-chest-xrays/data

---

## **Project Structure**

### **Classification Folder**
This folder contains the following models for classification tasks:
- **CheXNet Model**
- **DenseNet Model**
- **EfficientNet Model**
- **ResNet Model**

Additionally, it includes:
- **Ensemble Code**: To perform both average and weighted ensemble methods across the classification models.

---

### **Object Detection Folder**
This folder includes models for object detection tasks:
- **Faster R-CNN MODEL**:
- **YOLOv5 Config**: Custom configuration files for YOLOv5, including YAML file.

---

### **Components Folder**
Contains:
- **DataLoader for Classification Models**: Custom data loading pipeline for the classification models.
---
