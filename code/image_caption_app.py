import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load the model & processor
@st.cache_resource()
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# Streamlit UI
st.title("AI Image Caption Generator")
st.write("Upload an image, and the AI will generate a descriptive caption.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Generate caption
    with st.spinner("Generating caption..."):
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
    
    st.subheader("Generated Caption:")
    st.success(caption)

st.markdown("\n*Powered by BLIP Transformer Model*")


#Updated code for proposed model

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your model (adjust based on your implementation)
def load_model():
    model = tf.keras.models.load_model("your_model_path")
    return model

model = load_model()

# Image Preprocessing
def preprocess_image(image):
    image = image.resize((299, 299))  # Adjust based on model requirement
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to generate captions
def generate_caption(image=None, text_input=None):
    caption = "Generated Caption for the Image"  # Replace with actual model inference logic

    if text_input:
        caption += f" (Modified Based on Text Input: {text_input})"

    return caption

# Streamlit UI
st.title("AI Image Caption Generator")

# Upload Image
uploaded_file = st.file_uploader("Upload an Image:", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate Initial Caption
    if st.button("Generate Caption"):
        caption = generate_caption(image=image)
        st.session_state["generated_caption"] = caption  # Store for reuse
        st.write("Generated Caption:", caption)

# Text Input for Re-Generation
if "generated_caption" in st.session_state:
    user_input = st.text_input("Modify Caption using Text Input:")
    if user_input and st.button("Re-Generate Caption"):
        new_caption = generate_caption(image=image, text_input=user_input)
        st.write("New Caption:", new_caption)
