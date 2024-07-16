from fastapi import FastAPI
from fastapi import File, UploadFile
from fastapi.responses import Response
import io
import numpy as np
from pymupdf import Pixmap
from PIL import Image
import uvicorn

from core.model import inference_image
from utils.preprocess import preprocess_pdf
from utils.merge import merge_images_and_save_pdf

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/inference")
def inference(file: UploadFile = File(...)):
    file_obj = file.file.read()
    images:list[Pixmap] = preprocess_pdf(file_obj)
    final_results = []
    for pixmap in images:
        byte_code = pixmap.tobytes()
        image:Image = Image.open(io.BytesIO(byte_code))
        image = np.array(image)
        output_image = inference_image(image)
        final_results.append(output_image)
    merge_images_and_save_pdf(final_results)
    output_file = io.open("output.pdf", "rb")
    output_bytes = output_file.read()
    headers = {'Content-Disposition': 'attachment; filename="out.pdf"'}
    return Response(output_bytes, headers=headers, media_type='application/pdf')


if __name__ == "__main__":
    uvicorn.run("server:app",
        host="localhost",
        port=8000,
        reload=True
    )