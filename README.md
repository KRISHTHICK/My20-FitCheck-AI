# My20-FitCheck-AI
GenAI

Here’s a fresh **AI + Fashion project** idea with full code, explanation, and steps to run in **VS Code** and deploy to **GitHub**:

---

## 🧢 **Project Title**: **FitCheck AI - Outfit Rating & Style Recommender**

### 🎯 Objective:

Upload your outfit photo, and FitCheck AI gives:

* An **aesthetic/style label** (e.g., streetwear, formal, vintage)
* A **fashion score (1-10)** based on coordination, trendiness, and uniqueness
* **Style tips or improvement suggestions**
* Related outfit recommendations or hashtags

---

## 💡 Key Features:

* AI-powered outfit style detection
* Personalized suggestions to enhance your fashion
* Scoring system with explanation
* Ready-to-post fashion caption and hashtags

---

## 🛠️ Tech Stack:

* Python
* Streamlit
* Pretrained Vision Transformer (ViT) model from HuggingFace
* Local LLM via Ollama for generating suggestions

---

## 📁 Folder Structure:

```
FitCheck-AI/
├── app.py
├── requirements.txt
├── README.md
```

---

## 📦 `requirements.txt`

```txt
streamlit
torch
transformers
Pillow
langchain
langchain-community
```

---

## 🧠 `app.py` – Full Code

```python
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
```

---

## 🧾 `README.md`

````markdown
# 👗 FitCheck AI

**FitCheck AI** is your personal fashion critic. Upload your outfit photo and get:
- A style prediction
- A score out of 10 with fashion tips
- Matching accessories and hashtags

## 🔍 Features
- Predicts outfit aesthetic using Vision Transformer (ViT)
- Generates fashion advice and captions using local LLMs
- Works fully offline with Ollama

## 📦 Installation

```bash
git clone https://github.com/yourusername/FitCheck-AI.git
cd FitCheck-AI
pip install -r requirements.txt
````

## 🚀 Run the App

1. Start your LLM model:

```bash
ollama run tinyllama
```

2. Launch the app:

```bash
streamlit run app.py
```

## 🧠 Credits

* Vision Transformer: [nateraw/vit-fashion-classification](https://huggingface.co/nateraw/vit-fashion-classification)
* LLM: TinyLLaMA via Ollama

## 📸 Sample Inputs

* Try uploading casual, formal, or streetwear outfits for a fun rating!

```

---

Would you like to add outfit matching from your wardrobe or e-commerce links as a next step?

Let me know if you'd like a GitHub repository structure or push automation too!
```
