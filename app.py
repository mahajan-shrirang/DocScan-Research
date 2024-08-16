import os
import pandas as pd
import requests
import streamlit as st
import io
import fitz
from PIL import Image
from utils.preprocess import preprocess_pdf

st.set_page_config(page_title="DocScan", page_icon="📄", layout="wide")

BASEURL = "http://localhost:8000"

@st.experimental_dialog("Are you sure?")
def are_you_sure(selected_job):
    print(type(selected_job))
    st.header("Are you sure you want to kill job: " + str(selected_job))
    try:
        if st.button("Yes"):
            response = kill_job(selected_job)
            st.write(response)
            st.rerun()
    except ValueError as e:
        st.error(e)

def upload_file(file, model):
    url = BASEURL + ("/inference/" if model == "YOLOv4" else "/inference-detr/")
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(response.json())

def get_processes():
    url = BASEURL + "/jobs/"
    response = requests.get(url)
    return pd.DataFrame(response.json()["processes"])

def kill_job(job_id):
    url = BASEURL + "/kill/" + str(job_id) + "/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(response.text)

def download_file(job_id):
    url = BASEURL + "/get-output/" + str(job_id) + "/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise FileNotFoundError(response.text)
    
def display_pdf_preview(file):
    images = fitz.open(stream=file.read(), filetype="pdf")
    page = images.load_page(1)
    image = page.get_pixmap()
    img = Image.open(io.BytesIO(image.tobytes("png")))
    st.image(img, caption=f"Page 1", use_column_width=False , width=500)
    
st.title("DocScan")

select_model = st.sidebar.selectbox(
    "Select model", ("YoLo", "Detr"),
    placeholder="Select a model"
)

file = st.sidebar.file_uploader("Upload a file", type=["pdf"])

if file is not None:
    st.toast("File uploaded successfully",icon="🎉")
    if st.sidebar.button("Process"):
        display_pdf_preview(file)
        model = "YOLOv4" if select_model == "YoLo" else "DETR"
        try:
            response = upload_file(file, model)
            st.write(response)
        except ValueError as e:
            st.toast(e)
            
tab = st.tabs(["Get Processes"])[0]

if tab:
    jobs = get_processes()
    refresh = st.button("🔄️")
    display_pdf_preview(file)
    if refresh:
        jobs = get_processes()
        if jobs.shape[0] > 0:
            jobs['Timestamp'] = pd.to_datetime(jobs["Timestamp"])

    if jobs.shape[0] != 0:
        event = st.dataframe(jobs, on_select='rerun', hide_index=True, selection_mode="single-row")

        if len(event.selection['rows']):
            selected_row = event.selection['rows'][0]
            job_id = jobs.iloc[selected_row]['job_id']
            status = jobs.iloc[selected_row]['status']
            Timestamp = jobs.iloc[selected_row]['Timestamp']

            st.session_state['selected_job'] = job_id

            if status not in ["Completed", "Failed", "Killed"]:
                kill = st.button("❌ Kill Job")
                if kill:
                    are_you_sure(job_id)

            if status == "Completed":
                try:
                    st.download_button("Download Output Zip", download_file(job_id), "output.zip", "application/zip")
                except FileNotFoundError as e:
                    st.error(e)
    else:
        st.dataframe(pd.DataFrame(columns=["ID", "job_id", "status", "Timestamp"]))
