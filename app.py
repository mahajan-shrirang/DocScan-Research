from core.inference import inference_image
from utils.merge import merge_images_and_save_pdf
from utils.preprocess import preprocess_pdf
from utils.postprocess import get_summary

import streamlit as st
from pymupdf import Pixmap
import io
from PIL import Image
import numpy as np
import cv2


st.title("DocScan")

file = st.file_uploader("Upload a file", type=["pdf"])

if file:
    file_obj = file.getvalue()
    images: list[Pixmap] = preprocess_pdf(file_obj)
    st.success("Image extracted from PDF file")
    byte_code = images[0].tobytes()
    image1 = Image.open(io.BytesIO(byte_code))
    st.image(image1, caption="Sample Image", use_column_width=True)

    submit = st.button("Detect Objects")
    final_results = []
    if submit:
        output_images = []
        results = []
        for image in images:
            byte_code = image.tobytes()
            image: Image = Image.open(io.BytesIO(byte_code))
            image: cv2.Mat = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            output_image, boxes = inference_image(image)
            output_images.append(output_image)
            results.append(boxes)

        results = get_summary(results)
        results.index += 1
        st.table(results)
        merge_images_and_save_pdf(final_results, "output.pdf")
        output_file = io.open("output.pdf", "rb")
        st.download_button("Download PDF", output_file, "output.pdf")
        st.success("PDF file created with detected objects")
