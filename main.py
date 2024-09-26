if __name__ == "__main__":
    from ultralytics import YOLO
    model = YOLO("yolov8n.yaml")
    results = model.train(data="config.yaml", epochs=50)