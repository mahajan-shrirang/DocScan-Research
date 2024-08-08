import cv2
import datetime
from fastapi import FastAPI
from fastapi import File, UploadFile
from fastapi.responses import Response
import io
import numpy as np
import os
from pymupdf import Pixmap
from PIL import Image
import torch
import uvicorn

from core.detr import inference, load_checkpoint, load_model
from core.inference import inference_image
from utils.preprocess import preprocess_pdf
from utils.merge import merge_images_and_save_pdf

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/inference")
def inference_yolov4(file: UploadFile = File(...)):
    file_obj = file.file.read()
    images:list[Pixmap] = preprocess_pdf(file_obj)
    final_results = []
    for pixmap in images:
        byte_code = pixmap.tobytes()
        image:Image = Image.open(io.BytesIO(byte_code))
        image: cv2.Mat = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        output_image: cv2.Mat = inference_image(image)
        final_results.append(output_image)
    merge_images_and_save_pdf(final_results, "output.pdf")
    output_file = io.open("output.pdf", "rb")
    output_bytes = output_file.read()
    headers = {'Content-Disposition': 'attachment; filename="out.pdf"'}
    os.remove("output.pdf")
    return Response(output_bytes, headers=headers, media_type='application/pdf')


@app.post("/inference-detr")
def inference_detr(file: UploadFile = File(...)):
    file_obj = file.file.read()
    directory_name = f"{str(file.filename).split('.')[0]}-{datetime.datetime.now()}".replace(" ", "_").replace(":", "-").replace(".", "-")
    directory_name = "results/" + directory_name
    os.makedirs(directory_name, exist_ok=True)
    with open(f"{directory_name}/input.pdf", "wb") as f:
        f.write(file_obj)
    images:list[Pixmap] = preprocess_pdf(file_obj)
    for count, image in enumerate(images):
        byte_code = image.tobytes()
        image:Image = Image.open(io.BytesIO(byte_code))
        image.save(f"{directory_name}/page_{count}.png", "PNG")
    model, image_processor = load_model()
    model = load_checkpoint(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    id2labels = {0: "bar-scale", 1: "color stamp", 2: "detail label", 3: "north sign"}
    os.makedirs(f"{directory_name}/output", exist_ok=True)
    output_dir = f"{directory_name}/output/"
    results = inference(directory_name, 0.5, 0.5, model, image_processor, device, id2labels, output_dir)
    output_images = []
    for image in os.listdir(output_dir):
        print(image)
        img:cv2.Mat = cv2.imread(os.path.join(output_dir, image))
        output_images.append(img)
    merge_images_and_save_pdf(output_images, f"{directory_name}/output.pdf")
    output_file = io.open(os.path.join(directory_name, "output.pdf"), "rb")
    output_bytes = output_file.read()
    headers = {'Content-Disposition': 'attachment; filename="out.pdf"'}
    return Response(output_bytes, headers=headers, media_type='application/pdf')


if __name__ == "__main__":
    uvicorn.run("server:app",
        host="localhost",
        port=8000,
        reload=True
    )