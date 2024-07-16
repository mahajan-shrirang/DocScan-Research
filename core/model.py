from functools import lru_cache
import numpy as np
from ultralytics import YOLO

@lru_cache
def load_model() -> YOLO:
    return YOLO("runs/train/expv8/weights/best.pt")

def inference_image(image: np.ndarray) -> np.ndarray:
    results = yolo.predict(image)
    output_image = results[0].plot()
    return output_image

yolo = load_model()
