import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + tokenizer
model = DistilBertForSequenceClassification.from_pretrained(
    "ManoRakshithaa/factuality-distilbert"
)
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "ManoRakshithaa/factuality-distilbert"
)

model.to(DEVICE)
model.eval()

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.write("DistilBERT trained on WELFake → tested on ISOT")

text = st.text_area("Paste news article text here:")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()

        if pred == 1:
            st.error("🚨 Fake News 🚨")
        else:
            st.success("✅ Real News ✅")