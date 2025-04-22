import streamlit as st
from PIL import Image
import numpy as np

# --- Choose model type here ---
USE_BLIP_MODEL = True  # Set to False to use your custom TensorFlow model
# -------------------------------

# --- BLIP MODEL ---
if USE_BLIP_MODEL:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch

    @st.cache_resource()
    def load_blip_model():
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model

    processor, model = load_blip_model()

    def generate_blip_caption(image):
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption

# --- CUSTOM TF MODEL ---
else:
    import tensorflow as tf

    @st.cache_resource()
    def load_custom_model():
        model = tf.keras.models.load_model("your_model_path")  # Replace with actual model path
        return model

    model = load_custom_model()

    def preprocess_image(image):
        image = image.resize((299, 299))  # Change as per your model's input shape
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def generate_custom_caption(image=None, text_input=None):
        # Dummy placeholder - Replace with your model's prediction logic
        caption = "Generated Caption for the Image"
        if text_input:
            caption += f" (Modified Based on Text Input: {text_input})"
        return caption

# --- Streamlit UI ---
st.title("🖼 AI Image Caption Generator")
st.write("Upload an image and the AI will generate a caption for it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            if USE_BLIP_MODEL:
                caption = generate_blip_caption(image)
            else:
                caption = generate_custom_caption(image=image)
            st.session_state["generated_caption"] = caption
        st.success("Caption Generated!")
        st.subheader("Caption:")
        st.write(caption)

# Text input to modify caption
if "generated_caption" in st.session_state and not USE_BLIP_MODEL:
    user_input = st.text_input("Modify Caption using Text Input:")
    if user_input and st.button("Re-Generate Caption"):
        new_caption = generate_custom_caption(image=image, text_input=user_input)
        st.write("New Caption:", new_caption)

st.markdown("---")
st.markdown("Powered by BLIP Transformer / Custom AI Model")
