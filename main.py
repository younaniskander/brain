import streamlit as st
from utils import init_session_state_variables, dataset_unzip, rename_wrong_file, check_if_dataset_exists
from UNet_2D import init_model

def init_app():
    """
    App Configuration
    This functions sets & display the app title, its favicon, initialize some session_state values).
    It also verifies that the dataset exists in the environment and well unzipped.
    """

    # Set config and app title
    st.set_page_config(page_title="Image Segmentation", layout="wide", page_icon="🧠")
    st.title("Brain Tumors Segmentation 🧠")

    # Initialize session state variables
    init_session_state_variables()

    # File upload widget for dataset
    uploaded_file = st.file_uploader("Upload dataset (zip file)", type="zip")

    if uploaded_file is not None:
        # Save the uploaded file
        with open("uploaded_dataset.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Unzip the uploaded dataset
        dataset_unzip("uploaded_dataset.zip")

        # Rename the 355th file if necessary (it has a default incorrect name)
        rename_wrong_file("uploaded_dataset")

        # Check if the dataset exists in the environment to know if we can launch the app
        check_if_dataset_exists("uploaded_dataset")

        # Create & compile the CNN (U-Net model)
        model = init_model()
        
        return model

if __name__ == '__main__':
    model = init_app()
    if model is not None:
        launch_app(model)
    else:
        st.write("Please upload the dataset to proceed.")
