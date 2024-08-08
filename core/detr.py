import os 
import supervision as sv
from transformers import DetrForObjectDetection, DetrImageProcessor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import time
import torchvision
from torchvision.ops import box_iou
import torch
import pytorch_lightning
import cv2
import numpy as np
from collections import OrderedDict

MODEL_PATH = "D:\Data Science\DocScan-Research\Inference\DETR 11"
CHECKPOINT = "facebook/detr-resnet-50"
CHECKPOINT_PATH = "D:\Data Science\DocScan-Research\Inference\DETR 11\detr-epoch=99-val_loss=0.90.ckpt"
IMAGE_FOLDER = r"D:\Data Science\DocScan-Research\ExtractedImages2"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
id2labels = {0: "bar-scale", 1: "color stamp", 2: "detail label", 3: "north sign"}

def inference(image_folder, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, model, image_processor, device,id2labels):
    results_dict = {}
    for img in os.listdir(image_folder):
        IMAGE_PATH = os.path.join(image_folder, img)

        image = load_image(IMAGE_PATH)
        inputs = image_processor(images=image, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            target_sizes = torch.tensor([image.shape[:2]]).to(device)
            results = image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=CONFIDENCE_THRESHOLD,
                target_sizes=target_sizes
            )[0]

        detections = sv.Detections.from_transformers(transformers_results=results)
        
        box_annotator = sv.BoxAnnotator()
        frame = box_annotator.annotate(scene=image, detections=detections)
        image = Image.fromarray(frame)
        image_path = f"Temp3/results/annotated_{img}"
        all_labels = {0, 1, 2, 3}
        label = all_labels - set(detections.class_id)
        add_missing_label(image, image_path, label)
        results_dict[IMAGE_PATH.replace('Temp3/', '')] = results
    return results_dict
   
def load_model(MODEL_PATH, CHECKPOINT):
    model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
    return model, image_processor

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("model.model.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model

def move_model_to_device(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, device

def add_missing_label(image, save_path, labels):
    if labels:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text = f"Missing labels: {', '.join(map(str, labels))}"
        position = (10, 10)
        draw.text(position, text, fill="red", font=font)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)

def load_image(image_path):
    return cv2.imread(image_path)

def save_image(image, save_path):
    image.save(save_path)