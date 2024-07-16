import streamlit as st
from ultralytics import YOLO
from pymupdf import Pixmap
import io
from PIL import Image
import numpy as np

from core.model import inference_image
from utils.preprocess import preprocess_pdf
from utils.merge import merge_images_and_save_pdf

st.title("DocScan")

file = st.file_uploader("Upload a file", type=["pdf"])

if file:
    file_obj = file.getvalue()
    images:list[Pixmap] = preprocess_pdf(file_obj)
    st.success("Image extracted from PDF file")
    byte_code = images[0].tobytes()
    image1 = Image.open(io.BytesIO(byte_code))
    st.image(image1, caption="Sample Image", use_column_width=True)

    submit = st.button("Detect Objects")
    final_results = []
    if submit:
        for pixmap in images:
            byte_code = pixmap.tobytes()
            image:Image = Image.open(io.BytesIO(byte_code))
            image = np.array(image)
            output_image = inference_image(image)
            final_results.append(output_image)

        st.image(final_results[0], caption="Detected Objects", use_column_width=True)
        merge_images_and_save_pdf(final_results)
        output_file = io.open("output.pdf", "rb")
        st.download_button("Download PDF", output_file, "output.pdf")
        st.success("PDF file created with detected objects")