import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="DocScan",
    layout="wide",
    page_icon="📃"
)

BASEURL = "http://localhost:8000"

@st.experimental_dialog("Are you sure?")
def are_you_sure(job_id):
    st.header("Are you sure you want to kill job: " + str(job_id))
    try:
        if st.button("Yes"):
            response = kill_job(job_id)
            st.write(response)
            st.rerun()
    except ValueError as e:
        st.error(e)


def upload_file(file, model):
    if model == "YOLOv4":
        url = BASEURL + "/inference/"
    else:
        url = BASEURL + "/inference-detr/"

    files = {"file": (file.name, file.getvalue(), file.type)}
    response = requests.post(
        url,
        files=files
    )
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(response.json())


def get_processes():
    url = BASEURL + "/jobs/"
    response = requests.get(url)
    response = pd.DataFrame(response.json()["processes"])
    return response


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


st.title("DocScan")

try:
    jobs = get_processes()
except Exception:
    st.error("It seems the backend is not working properly. Try fixing the issues with the backend.")
    st.stop()

tab = st.tabs([
    "Input Data",
    "Ongoing Process"
])

with tab[0]:
    file = st.file_uploader("Upload a file", type=["pdf"])

    model = st.selectbox("Select Model",
        options=[
            "YOLOv4",
            "DETR"
        ],
        index=None,
        placeholder="Select a model"
    )

    if file is not None:
        if st.button("Predict"):
            try:
                response = upload_file(file, model)
                st.success("Job Started Successfully")
                st.write(response)

            except ValueError as e:
                st.error(e)

with tab[1]:
    jobs = get_processes()

    refresh = st.button("🔄️")
    if refresh:
        jobs = get_processes()
        if jobs.shape[0] > 0:
            jobs['Timestamp'] = pd.to_datetime(jobs["Timestamp"])
            # jobs["file_path"] = jobs['file_path']

    if jobs.shape[0] != 0:
        event = st.dataframe(jobs, on_select='rerun', hide_index=True, selection_mode="single-row")

        if len(event.selection['rows']):
            selected_row = event.selection['rows'][0]
            job_id = jobs.iloc[selected_row]['job_id']
            status = jobs.iloc[selected_row]['status']
            Timestamp = jobs.iloc[selected_row]['Timestamp']

            st.session_state['selected_job'] = job_id
            st.subheader("Selected Job: "+job_id)

            if status not in ["Completed", "Failed", "Killed"]:
                st.button("❌ Kill Job", on_click=are_you_sure, args=(job_id,))

            if status == "Completed":
                try:
                    st.download_button("Download Output Zip", download_file(job_id), "output.zip", "application/zip")
                except FileNotFoundError as e:
                    st.error(e)
    else:
        st.dataframe(pd.DataFrame(columns=["ID", "job_id", "status", "Timestamp"]))
