from ultralytics import YOLO

# model = YOLO("yolov5n.pt")
model = YOLO("yolov8n.pt")

results = model.train(data="./data/annotation.yaml", epochs=20, imgsz=640, batch=32, project="runs/train", name="expv8_2", exist_ok=True)