from core.inference import inference_image
from utils.merge import merge_images_and_save_pdf
from utils.preprocess import preprocess_pdf
from utils.postprocess import get_summary

import streamlit as st
from pymupdf import Pixmap
import io
import os
import datetime
from PIL import Image
import numpy as np
import cv2
import time
from zipfile import ZipFile


st.title("DocScan")

file = st.file_uploader("Upload a file", type=["pdf"])
os.makedirs("results", exist_ok=True)

if file:
    if not file.name.endswith(".pdf"):
        st.error("Invalid file format. Please upload a PDF file.")
    if st.button("Process"):
        file_obj = file.getvalue()
        directory_name = f"{str(file.name).split('.')[0]}-\
                {datetime.datetime.now()}".replace(" ", "_").replace(":", "-").replace(".", "-")
        directory_name = "results/" + directory_name
        os.makedirs(directory_name, exist_ok=True)
        with open(f"{directory_name}/input.pdf", "wb") as f:
            f.write(file_obj)
        os.makedirs(f"{directory_name}/output", exist_ok=True)
        output_dir = f"{directory_name}/output"
        images: list[Pixmap] = preprocess_pdf(file_obj)
        if len(images) == 0:
            st.error("No images found in the PDF file.")
        output_images = []
        results = []
        start_time = time.time()
        for image in images:
            byte_code = image.tobytes()
            image: Image = Image.open(io.BytesIO(byte_code))
            image: cv2.Mat = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            output_image, boxes = inference_image(image)
            output_images.append(output_image)
            results.append(boxes)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        st.write(f"Time taken: {end_time - start_time} seconds")
        merge_images_and_save_pdf(output_images,
                                    save_path=f"{output_dir}/output.pdf")
        if len(results[0]) == 0:
            st.error("No objects detected in the PDF file.")
        results = get_summary(results)
        results.index += 1
        results.to_csv(f"{output_dir}/summary.csv", index_label="page no.")
        print(f"Results saved to `{output_dir}/summary.csv`")
        print(f"Output saved to `{output_dir}/output.pdf`")
        with ZipFile(f"{output_dir}/output.zip", 'w') as zipf:
            zipf.write(f"{output_dir}/output.pdf", "output.pdf")
            zipf.write(f"{output_dir}/summary.csv", "summary.csv")
        output_file = io.open(os.path.join(output_dir, "output.zip"), "rb")
        output_bytes = output_file.read()
        st.download_button("Download output", output_bytes, "output.zip", "application/zip")
