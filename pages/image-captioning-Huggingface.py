import torch
import time
import streamlit as st
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration


@st.cache(allow_output_mutation=True)
def get_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model =get_model()
# unconditional image captioning

st.title("Image Captioning App")
# define the layout of your app
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
submit_button = st.button("Compute")

if uploaded_file is not None:
    if submit_button :
        # Load the uploaded image
        image = Image.open(uploaded_file).convert('RGB')

        # Use the pre-trained model to generate a caption for the uploaded image
        progress_text = "Operation in progress. Please wait."
        bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs)
            time.sleep(0.1)
            bar.progress(percent_complete + 1, text=progress_text)
            
        # Display the uploaded image and its generated caption
        st.image(image)
        st.write("Generated Caption:")
        st.write(processor.decode(out[0], skip_special_tokens=True))
        time.sleep(5)
        st.success('Congratulations task is done ', icon="✅")
        st.balloons()
    else:
      st.error('Error , Plz..... press Compute', icon="🚨")
