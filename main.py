import streamlit as st
import os
from interface_tumor import *
from utils import init_session_state_variables, dataset_unzip, rename_wrong_file, check_if_dataset_exists
from UNet_2D import init_model
from variables import data_path

def upload_dataset():
    """
    Function to upload dataset from user's PC.
    """
    st.warning("Please upload the dataset file (ZIP format).")
    uploaded_file = st.file_uploader("Upload Dataset", type=["zip"])

    if uploaded_file is not None:
        with open(os.path.join(data_path, "uploaded_dataset.zip"), "wb") as f:
            f.write(uploaded_file.read())
        st.success("Dataset uploaded successfully!")

def init_app():
    """
    App Configuration
    This function sets & displays the app title, its favicon, initializes some session_state values.
    It also verifies that the dataset exists in the environment and is well unzipped.
    """

    # Set config and app title
    st.set_page_config(page_title="Image Segmentation", layout="wide", page_icon="🧠")
    st.title("Brain Tumors Segmentation 🧠")

    # Initialize session state variables
    init_session_state_variables()

    # Upload dataset if not already uploaded
    if not check_if_dataset_exists():
        upload_dataset()
    else:
        st.info("Dataset already exists.")

    # Unzip dataset if not already done
    dataset_unzip()

    # Rename the 355th file if necessary (it has a default incorrect name)
    rename_wrong_file(data_path)

    # Create & compile the CNN (U-Net model)
    model = init_model()

    return model

if __name__ == '__main__':
    model = init_app()
    launch_app(model)
