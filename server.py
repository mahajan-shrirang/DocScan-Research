from core.detr import inference, load_checkpoint, load_model
from core.inference import inference_image
from core.db import (
    insert_process_details_into_db,
    create_processes_table,
    update_process_status,
    update_process_id,
    get_data_from_db,
    get_db_conn,
    get_processes,
    get_process_details,
)
from utils.postprocess import get_summary
from utils.preprocess import preprocess_pdf
from utils.merge import merge_images_and_save_pdf
from utils.logger import log_message

import cv2
import datetime
from fastapi import FastAPI
from fastapi import File, UploadFile
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import numpy as np
import multiprocessing
import os
from pymupdf import Pixmap
from PIL import Image
import time
import torch
import uvicorn
import warnings
from zipfile import ZipFile

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

conn = get_db_conn()

create_processes_table(conn)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def inference_yolo(directory_name: str, job_id: str):
    try:
        update_process_status(job_id, "In Progress")
        os.makedirs(f"{directory_name}/output", exist_ok=True)
        output_dir = f"{directory_name}/output"
        file = open(f"{directory_name}/input.pdf", "rb")
        file_obj = file.read()
        file.close()
        images: list[Pixmap] = preprocess_pdf(file_obj)
        if len(images) == 0:
            log_message("No images found in the PDF file. Job id: " + job_id, 1)
            log_message("Returning response with status code 400. Job id: " + job_id, 0)
            update_process_status(job_id, "Failed")
        output_images = []
        results = []
        start_time = time.time()
        log_message("Starting inference.", 0)
        for image in images:
            byte_code = image.tobytes()
            image: Image = Image.open(io.BytesIO(byte_code))
            image: cv2.Mat = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            output_image, boxes = inference_image(image)
            output_images.append(output_image)
            results.append(boxes)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        log_message("Inference completed. Job id: " + job_id, 0)
        merge_images_and_save_pdf(output_images,
                                  save_path=f"{output_dir}/output.pdf")
        log_message("Merging images and saving as PDF. Job id: " + job_id, 0)
        if len(results[0]) == 0:
            log_message("No images found in the PDF file. Job id: " + job_id, 1)
            log_message("Returning response with status code 400. Job id: " + job_id, 0)
            update_process_status(job_id, "Failed")
        results = get_summary(results)
        results.index += 1
        results.to_csv(f"{output_dir}/summary.csv", index_label="page no.")
        print(f"Results saved to `{output_dir}/summary.csv`")
        print(f"Output saved to `{output_dir}/output.pdf`")
        log_message("Saved results CSV and output PDF to the output folder. Job id: " + job_id, 0)
        with ZipFile(f"{output_dir}/output.zip", 'w') as zipf:
            zipf.write(f"{output_dir}/output.pdf", "output.pdf")
            zipf.write(f"{output_dir}/summary.csv", "summary.csv")
        log_message("Returning response with status code 200. Job id: " + job_id, 0)
        update_process_status(job_id, "Completed")

    except Exception as e:
        log_message(f"Error occurred: {str(e)}", 1)
        return JSONResponse(content={"message": str(e)}, status_code=400)


@app.get("/")
async def read_root():
    """Root endpoint

    Returns:
        dict: Hello World
    """
    return {"Hello": "World"}


@app.post("/inference")
async def inference_yolov4(file: UploadFile = File(...)):
    """Inference using YOLOv4 model

    Args:
        file (UploadFile, optional): PDF file to be processed. Defaults to
                                     File(...).

    Returns:
        Response: Response object containing the output zip file
    """
    log_message("Received a request for inference using YOLOv4 model. File: "
                + file.filename, 0)
    try:
        if file.content_type != "application/pdf":
            log_message("Invalid file type.", 1)
            log_message("Returning response with status code 400.", 0)
            return JSONResponse(content={"message": "Invalid file type.\
                                        Please upload a PDF file."},
                                status_code=400)
        file_obj = file.file.read()
        directory_name = f"{str(file.filename).split('.')[0]}-\
            {datetime.datetime.now()}".replace(" ", "_")\
            .replace(":", "-").replace(".", "-")
        directory_name = os.path.join("results", directory_name)
        os.makedirs(directory_name, exist_ok=True)
        log_message(f"Saving the input file to `{directory_name}/input.pdf`",
                    level=0)
        with open(f"{directory_name}/input.pdf", "wb") as f:
            f.write(file_obj)
        job_id = insert_process_details_into_db("", directory_name)
        process = multiprocessing.Process(target=inference_yolo,
                                          args=(directory_name, job_id))
        process.start()
        process_id = str(process.pid)
        update_process_id(job_id, process_id)
        return JSONResponse(content={"message": "Processing started.",
                                     "job_id": job_id}, status_code=200)
    except Exception as e:
        log_message(f"Error occurred: {str(e)}", 1)
        return JSONResponse(content={"message": str(e)}, status_code=400)


@app.post("/inference-detr")
async def inference_detr(file: UploadFile = File(...)):
    """Inference using DETR model

    Args:
        file (UploadFile, optional): PDF file to be processed. Defaults to
                                     File(...).

    Returns:
        Response: Response object containing the output zip file
    """
    log_message("Received a request for inference using DETR model. File: "
                + file.filename, 0)
    if file.content_type != "application/pdf":
        log_message("Invalid file type.", 1)
        log_message("Returning response with status code 400.", 0)
        return JSONResponse(content={"message": "Invalid file type. Please \
                                     upload a PDF file."}, status_code=400)
    file_obj = file.file.read()
    directory_name = f"{str(file.filename).split('.')[0]}-\
        {datetime.datetime.now()}".replace(" ", "_").replace(":", "-")\
        .replace(".", "-")
    directory_name = "results/" + directory_name
    os.makedirs(directory_name, exist_ok=True)
    log_message(f"Saving the input file to `{directory_name}/input.pdf`", 0)
    with open(f"{directory_name}/input.pdf", "wb") as f:
        f.write(file_obj)
    output_dir = f"{directory_name}/output"
    os.makedirs(output_dir, exist_ok=True)
    images: list[Pixmap] = preprocess_pdf(file_obj)
    if len(images) == 0:
        log_message("No images found in the PDF file.", 1)
        log_message("Returning response with status code 400.", 0)
        return JSONResponse(content={"message": "No images found in the PDF \
                                     file."}, status_code=400)
    model, image_processor = load_model()
    model = load_checkpoint(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    id2labels = {
        0: "bar-scale",
        1: "color stamp",
        2: "detail label",
        3: "north sign"
    }
    output_images = []
    results = []
    start_time = time.time()
    log_message("Starting inference.", 0)
    for image in images:
        image = Image.open(io.BytesIO(image.tobytes()))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        output_image, results_dict = inference(image, 0.5, 0.5, model,
                                               image_processor, device,
                                               id2labels, output_dir)
        output_image = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
        output_images.append(output_image)
        results.append(results_dict['labels'].tolist())
    end_time = time.time()
    log_message("Inference completed.", 0)
    print(f"Time taken: {end_time - start_time} seconds")
    if len(results[0]) == 0:
        log_message("No images found in the PDF file.", 1)
        log_message("Returning response with status code 400.", 0)
        return JSONResponse(content={"message": "No objects detected in the "
                                     + "PDF file."},
                            status_code=400)
    merge_images_and_save_pdf(output_images,
                              save_path=f"{output_dir}/output.pdf")
    log_message("Merging images and saving as PDF.", 0)
    results = get_summary(results)
    results.index += 1
    results.to_csv(f"{output_dir}/summary.csv", index_label="page no.")
    print(f"Results saved to `{output_dir}/summary.csv`")
    print(f"Output saved to `{output_dir}/output.pdf`")
    log_message("Saved results CSV and output PDF to the output folder.", 0)
    with ZipFile(f"{output_dir}/output.zip", 'w') as zipf:
        zipf.write(f"{output_dir}/output.pdf", "output.pdf")
        zipf.write(f"{output_dir}/summary.csv", "summary.csv")
    output_file = io.open(os.path.join(output_dir, "output.zip"), "rb")
    output_bytes = output_file.read()
    headers = {'Content-Disposition': 'attachment; filename="output.zip"'}
    log_message("Returning response with status code 200.", 0)
    return Response(output_bytes, headers=headers,
                    media_type='application/pdf', status_code=200)


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get the status of a job

    Args:
        job_id (str): Job ID

    Returns:
        dict: Status of the job
    """
    log_message(f"Received a request to get status of job with ID: {job_id}",
                0)
    status = get_process_details(job_id)
    log_message("Returning response with status code 200.", 0)
    return JSONResponse(content=status, status_code=200)


@app.get("/kill/{job_id}")
async def kill_job(job_id: str):
    """Kill a job

    Args:
        job_id (str): Job ID

    Returns:
        dict: Message
    """
    log_message(f"Received a request to kill job with ID: {job_id}", 0)
    try:
        status = get_process_details(job_id)
        process_id = status["process_id"]
        os.kill(int(process_id), 9)
        update_process_status(job_id, "Killed")
        log_message("Job killed successfully.", 0)
        log_message("Returning response with status code 200.", 0)
        return JSONResponse(content={"message": "Job killed successfully."},
                            status_code=200)
    except Exception as e:
        log_message(f"Error occurred: {str(e)}", 1)
        return JSONResponse(content={"message": str(e)}, status_code=400)


@app.get("/get-output/{job_id}")
async def get_output(job_id: str):
    """Get the output of a job

    Args:
        job_id (str): Job ID

    Returns:
        Response: Output zip file
    """
    log_message(f"Received a request to get output of job with ID: {job_id}", 0)
    status = get_process_details(job_id)
    if status["status"] == "Completed":
        output_dir = status["file_path"]
        output_file = io.open(os.path.join(output_dir, "output", "output.zip"), "rb")
        output_bytes = output_file.read()
        headers = {'Content-Disposition': 'attachment; filename="output.zip"'}
        log_message("Returning response with status code 200.", 0)
        return Response(output_bytes, headers=headers,
                        media_type='application/pdf', status_code=200)
    else:
        log_message("Job not completed yet.", 1)
        log_message("Returning response with status code 400.", 0)
        return JSONResponse(content={"message": "Job not completed yet."},
                            status_code=400)


if __name__ == "__main__":
    uvicorn.run("server:app", host="localhost", port=8000, reload=True)
