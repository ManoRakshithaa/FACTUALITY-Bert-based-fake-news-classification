import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Factuality Detector",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Factuality Detector")
st.write("This model predicts whether a statement is factual or not.")

# ---------------------------
# Load Model from HuggingFace
# ---------------------------

@st.cache_resource
def load_model():
    model_name = "ManoRakshitha/factuality-distilbert"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        subfolder="saved_model"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        subfolder="saved_model"
    )

    return tokenizer, model


tokenizer, model = load_model()

# ---------------------------
# User Input
# ---------------------------

user_input = st.text_area("Enter a statement:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        # Assuming:
        # 0 = Not Factual
        # 1 = Factual
        if prediction == 1:
            st.success("✅ This statement appears to be FACTUAL.")
        else:
            st.error("❌ This statement appears to be NOT FACTUAL.")