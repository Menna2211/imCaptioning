import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

st.title("Welcome to My TXImg! 👋")

st.markdown(
    """
    TXImg is an open-source app built specifically for Image Captioning and Stable Diffusion projects.
    ### Image Caption:
    The application allows users to upload an image and generate a descriptive caption for the image Using:
    - Hugging Face Model: [blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
    - Github model: [CATR](https://github.com/saahiluppal/catr)
"""
)
