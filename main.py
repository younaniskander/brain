import streamlit as st
import subprocess
from utils import init_session_state_variables, rename_wrong_file, check_if_dataset_exists
from UNet_2D import init_model

# GLOBAL VARIABLES
IMG_SIZE = 128

# SLICES RANGE (for predicted segmentation & original one / ground truth)
VOLUME_START_AT = 0
VOLUME_SLICES = 155

# Sorted list of our test patients (patients that have not been used for the model training part)
samples_test = ['BraTS20_Training_009', 'BraTS20_Training_013', 'BraTS20_Training_017', 'BraTS20_Training_019',
                'BraTS20_Training_025', 'BraTS20_Training_037', 'BraTS20_Training_040', 'BraTS20_Training_041',
                'BraTS20_Training_051', 'BraTS20_Training_054', 'BraTS20_Training_056', 'BraTS20_Training_072',
                'BraTS20_Training_076', 'BraTS20_Training_077', 'BraTS20_Training_082', 'BraTS20_Training_083',
                'BraTS20_Training_094', 'BraTS20_Training_095', 'BraTS20_Training_096', 'BraTS20_Training_107',
                'BraTS20_Training_112', 'BraTS20_Training_113', 'BraTS20_Training_122', 'BraTS20_Training_129',
                'BraTS20_Training_146', 'BraTS20_Training_160', 'BraTS20_Training_167', 'BraTS20_Training_180',
                'BraTS20_Training_185', 'BraTS20_Training_199', 'BraTS20_Training_201', 'BraTS20_Training_222',
                'BraTS20_Training_237', 'BraTS20_Training_242', 'BraTS20_Training_249', 'BraTS20_Training_255',
                'BraTS20_Training_266', 'BraTS20_Training_278', 'BraTS20_Training_292', 'BraTS20_Training_297',
                'BraTS20_Training_302', 'BraTS20_Training_324', 'BraTS20_Training_325', 'BraTS20_Training_335',
                'BraTS20_Training_356']

# Add a Random patient choice to this list
samples_test.insert(0, "Random patient")

# Dictionary which links the modalities to their file names in the database
modalities_dict = {
    '_t1.nii': 'T1',
    '_t1ce.nii': 'T1CE',
    '_t2.nii': 'T2',
    '_flair.nii': 'FLAIR'}

def init_app():
    """
    App Configuration
    This function sets and displays the app title, its favicon, initializes some session_state values,
    verifies that the dataset exists in the environment, and is well unzipped.
    """
    # Set config and app title
    st.set_page_config(page_title="Image Segmentation", layout="wide", page_icon="🧠")
    st.title("Brain Tumors Segmentation 🧠")

    # Initialize session state variables
    init_session_state_variables()

    # Run Kaggle command to download the dataset
    subprocess.run(["kaggle", "datasets", "download", "-d", "awsaf49/brats20-dataset-training-validation"])

    # # Rename the 355th file if necessary (it has a default incorrect name)
    # rename_wrong_file(output_path)

    # # Check if the dataset exists in the environment to know if we can launch the app
    # check_if_dataset_exists(output_path)

    # Create & compile the CNN (U-Net model)
    model = init_model()
        
    return model

if __name__ == '__main__':
    model = init_app()
    if model is not None:
        launch_app(model)
    else:
        st.write("Please make sure the dataset is downloaded from Kaggle to proceed.")
