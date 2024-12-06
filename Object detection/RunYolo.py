from yolov5 import train, val
def main():

    # Define the path to your YAML configuration file
    data_yaml = r'C:\Users\henni\PycharmProjects\IKT450\Project\Yolo_Components\chest_xray.yaml'
    weights_path = 'yolov5s.pt'  # Pretrained weights to start with
    weights_path_validation = r'C:\Users\henni\PycharmProjects\IKT450\Project\YOLO_Components\runs\train\exp4\weights/best.pt'
    img_size = 640
    batch_size = 16
    epochs = 50
    # Training

    #print("Starting training...")
   # train.run(img=img_size, batch=batch_size, epochs=epochs, data=data_yaml, weights=weights_path)

    # Validation
    print("Starting validation...")
    val.run(data=data_yaml, imgsz=img_size, weights=weights_path_validation)

if __name__ == "__main__":
    main()
