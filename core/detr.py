import os
import supervision as sv
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
import json
from collections import OrderedDict
import functools


with open(r"config.json") as fp:
    configs = json.load(fp)
    detr_model_path = configs['DETR_MODEL_PATH']
    detr_checkpoint = configs['DETR_CHECKPOINT_PATH']

CHECKPOINT = "facebook/detr-resnet-50"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
id2labels = {
    0: "bar-scale",
    1: "color stamp",
    2: "detail label",
    3: "north sign"
}


def inference(image, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, model,
              image_processor, device, id2labels, save_path):
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
    return image, results


@functools.lru_cache(maxsize=1)
def load_model():
    model = DetrForObjectDetection.from_pretrained(detr_model_path)
    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
    return model, image_processor


def load_checkpoint(model):
    checkpoint = torch.load(detr_checkpoint, map_location='cpu')
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
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    position = (10, 10)
    draw.text(position, '', fill="red", font=font)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)


def load_image(image_path):
    return cv2.imread(image_path)


def save_image(image, save_path):
    image.save(save_path)
