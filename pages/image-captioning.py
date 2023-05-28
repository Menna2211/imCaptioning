from transformers import BertTokenizer
import torch
import time
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)
    caption_template[:, 0] = start_token
    mask_template[:, 0] = False
    return caption_template, mask_template

caption, cap_mask = create_caption_and_mask(start_token, 128)

# Model 1
@st.cache_resource(show_spinner=False ,ttl=3600) 
def get_model1():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model1 = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model1
  
processor, model1 =get_model1()

# Model 2 
@st.cache_resource(show_spinner=False ,ttl=3600) 
def get_model2():
    model2 = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)  # you can choose between v1, v2 and v3
    return  model2

model2 =get_model2()

st.title("Image Captioning App")
# define the layout of your app
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
model = st.selectbox("Select a Model", ["Select a Model","Hugging-Face", "Github"])
submit_button = st.button("Compute")
if model == "Select a Model" and not submit_button :
  st.stop()
elif model == "Select a Model" and  submit_button :
  st.warning('Warning.....!!,Plz..... Select a Model ', icon="⚠️")

if model == "Hugging-Face":
    if uploaded_file is not None :
        # Load the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image)
        # Use the pre-trained model to generate a caption for the uploaded image
        progress_text = "Operation in progress. Please wait."
        bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            inputs = processor(image, return_tensors="pt")
            out = model1.generate(**inputs , max_new_tokens=100)
            time.sleep(0.1)
            bar.progress(percent_complete + 1, text=progress_text)

        # Display the uploaded image and its generated caption
        st.write("Generated Caption:")
        st.write(processor.decode(out[0], skip_special_tokens=True))
        time.sleep(5)
        st.success('Congratulations..!! task is done ', icon="✅")
        st.balloons()
    else:
      st.error('Error...!!,Plz..... Upload an image' , icon="🚨")

elif model == "Github":
    if uploaded_file is not None :
        # Load the uploaded image
        im = Image.open(uploaded_file)
        st.image(im)
        # Preprocess the input image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.ToTensor(),         # Convert the image to a tensor
            transforms.Normalize(          # Normalize the image
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
        image = transform(im).unsqueeze(0)  # Add a batch dimension
        #@torch.no_grad()
        def evaluate():
            for i in range(128-1):
                predictions = model2(image, caption, cap_mask)
                predictions = predictions[:, i, :]
                predicted_id = torch.argmax(predictions, axis=-1)

                if predicted_id[0] == 102:
                    return caption
                caption[:, i+1] = predicted_id[0]
                cap_mask[:, i+1] = False

            return caption

        # Use the pre-trained model to generate a caption for the uploaded image
        progress_text = "Operation in progress. Please wait."
        bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            output = evaluate()
            time.sleep(0.1)
            bar.progress(percent_complete + 1, text=progress_text)

        # Display the uploaded image and its generated caption
        st.write("Generated Caption:")
        result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        st.write(result.capitalize())
        time.sleep(5)
        st.success('Congratulations...!! task is done ', icon="✅")
        st.balloons()
    else:
       st.error('Error...!!,Plz..... Upload an image' , icon="🚨")
