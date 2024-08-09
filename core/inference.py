from core.yolov4.model import Darknet
from core.yolov4.torch_utils import do_detect
import torch
import cv2
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    model = Darknet('core/yolov4/yolov4-custom.cfg', inference=True)
    print(model.width, model.height)
    model.load_state_dict(torch.load('core/yolov4/yolov4.pt',
                                     map_location=device, weights_only=False))
    model.eval()
    return model


def predict(model: Darknet, img: cv2.Mat):
    temp_img = img.copy()
    temp_img = cv2.resize(temp_img, (model.width, model.height))
    boxes = do_detect(model, temp_img, 0.4, 0.6, use_cuda=0)
    arr = np.array(boxes[0])
    if len(arr) == 0:
        return [], []
    labels = arr[:, 6].astype(int).tolist()
    return boxes[0], labels


def plot_boxes(boxes, img):
    img = np.copy(img)
    width = img.shape[1]
    height = img.shape[0]

    class_colors = {
        0: (255, 0, 255),
        1: (0, 0, 255),
        2: (0, 255, 255),
        3: (0, 255, 0)
    }
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        img = cv2.rectangle(img, (x1, y1), (x2, y2),
                            color=class_colors[int(box[6])], thickness=2)

    return img


def inference_image(img: cv2.Mat):
    model = load_model()
    boxes, labels = predict(model, img)

    img = plot_boxes(boxes, img)
    return img, labels
