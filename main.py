import streamlit as st
import kaggle
import os
from interface_tumor import *
from utils import init_session_state_variables, dataset_unzip, rename_wrong_file, check_if_dataset_exists
from UNet_2D import init_model
from variables import data_path

def download_kaggle_dataset(dataset_name):
    """
    Function to download a Kaggle dataset using the Kaggle API.
    """
    # Authenticate with the Kaggle API
    kaggle.api.authenticate()

    # Download dataset
    kaggle.api.dataset_download_files(dataset_name, path=data_path, unzip=True)

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

    # Download Kaggle dataset if not already downloaded
    if not check_if_dataset_exists():
        st.warning("Downloading dataset from Kaggle...")
        download_kaggle_dataset("awsaf49/brats20-dataset-training-validation")
        st.success("Dataset downloaded successfully!")

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
