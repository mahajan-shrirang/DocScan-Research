from ultralytics import YOLO

# model = YOLO("yolov5n.pt")
model = YOLO("yolov8n.pt")

results = model.train(data="./data/data/annotation.yaml", epochs=10, imgsz=640, batch=16, project="runs/train", name="expv8", exist_ok=True)