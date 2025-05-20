import streamlit as st
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
import os

# Set up Streamlit page
st.set_page_config(page_title="👗 FitCheck AI", layout="centered")
st.title("👗 FitCheck AI")
st.subheader("Upload your outfit and get rated like a fashionista!")

# Load ViT model
@st.cache_resource
def load_model():
    model = ViTForImageClassification.from_pretrained("nateraw/vit-fashion-classification")
    extractor = ViTFeatureExtractor.from_pretrained("nateraw/vit-fashion-classification")
    return model, extractor

model, extractor = load_model()

# Upload outfit photo
uploaded_file = st.file_uploader("📸 Upload your outfit photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Outfit", use_column_width=True)

    # Extract features
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = outputs.logits.argmax(-1).item()
        label = model.config.id2label[predicted_class]

    st.markdown(f"### 🧵 Predicted Style: `{label}`")

    # LLM-based scoring and tips
    llm = Ollama(model="tinyllama")

    score_prompt = PromptTemplate(
        input_variables=["label"],
        template="Rate an outfit with style '{label}' on a scale of 1 to 10 and explain why. Suggest 1 improvement and 2 matching accessories."
    )

    caption_prompt = PromptTemplate(
        input_variables=["label"],
        template="Write a trendy social media caption and 3 fashion hashtags for an outfit with '{label}' aesthetic."
    )

    score_chain = LLMChain(llm=llm, prompt=score_prompt)
    caption_chain = LLMChain(llm=llm, prompt=caption_prompt)

    if st.button("🧠 Rate My Fit"):
        with st.spinner("Analyzing your outfit..."):
            score_result = score_chain.run(label=label)
            caption_result = caption_chain.run(label=label)

        st.markdown("### 🎯 Fashion Score & Tips:")
        st.info(score_result)

        st.markdown("### 📝 Caption & Hashtags:")
        st.success(caption_result)
