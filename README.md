Factuality — BERT-Based Fake News Classification

<p align="center">
<b>Real-time Fake News Detection using Fine-Tuned DistilBERT</b>




Built with Transformers • PyTorch • Streamlit • HuggingFace
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" alt="Python">
<img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface" alt="Transformers">
<img src="https://img.shields.io/badge/PyTorch-DeepLearning-red?logo=pytorch" alt="PyTorch">
<img src="https://img.shields.io/badge/Deployed-Streamlit-red?logo=streamlit" alt="Streamlit">
</p>

🌐 Live Demo

🔗 Try the App Here: 👉 https://YOUR_STREAMLIT_LINK_HERE

🧠 Project Overview

Factuality is a deep learning-based web application designed to combat misinformation by classifying news text as:

✅ Real

❌ Fake

The system utilizes a fine-tuned DistilBERT transformer model, trained on a comprehensive Kaggle fake news dataset. This project serves as a full-cycle demonstration of modern NLP workflows.

Key Highlights:

NLP Preprocessing: Advanced cleaning and tokenization.

Transformer Fine-tuning: Optimization of DistilBERT for binary classification.

Seamless Deployment: Model hosting on HuggingFace Hub and frontend on Streamlit Cloud.

🏗 Tech Stack

Category

Technology

Language

Python 3.10

ML Framework

PyTorch, HuggingFace Transformers

Model Architecture

DistilBERT (distilbert-base-uncased)

Web Framework

Streamlit

Data Source

Kaggle

Model Hosting

HuggingFace Model Hub

📚 Dataset

The model was trained on a publicly available Fake News dataset from Kaggle containing labeled real and fake news articles.

Preprocessing Pipeline:

Text Cleaning: Removal of special characters and noise.

Tokenization: Using the specialized DistilBERT tokenizer.

Standardization: Padding and truncation to a max_length of 256.

Training: Fine-tuned using CrossEntropyLoss on a GPU-enabled environment.

[!IMPORTANT]

Dataset files are not included in this repository due to licensing restrictions.

🏋️ Model Training Details

The core of this project is a distilbert-base-uncased model fine-tuned for sequence classification.

Optimizer: AdamW

Loss Function: CrossEntropyLoss

Library: PyTorch + HuggingFace Transformers

View the model on HuggingFace Hub: 👉 ManoRakshitha/factuality-distilbert

📂 Project Structure

FACTUALITY-Bert-based-fake-news-classification
┣ 📜 app.py              # Streamlit frontend & inference logic
┣ 📜 requirements.txt    # Project dependencies
┗ 📜 README.md           # Documentation


The model weights are loaded dynamically from the HuggingFace Hub at runtime to keep the repository lightweight.

⚙️ Installation (Run Locally)

1️⃣ Clone the repository

git clone [https://github.com/ManoRakshithaa/FACTUALITY-Bert-based-fake-news-classification.git](https://github.com/ManoRakshithaa/FACTUALITY-Bert-based-fake-news-classification.git)
cd FACTUALITY-Bert-based-fake-news-classification


2️⃣ Install dependencies

pip install -r requirements.txt


3️⃣ Run the Streamlit app

streamlit run app.py


🎯 Features

Real-time Classification: Get instant results upon pasting news content.

Clean UI: Minimalist and intuitive user interface.

High Performance: Powered by a lightweight DistilBERT architecture for fast inference.

Hardware Agnostic: Automatically detects and uses GPU if available, otherwise defaults to CPU.

🚀 Deployment Architecture

Model Hosting: Weights and configuration reside on the HuggingFace Model Hub.

Frontend: Streamlit Cloud pulls the code from GitHub.

Inference: When a user submits text, the app fetches the model via the Transformers API for local inference on the Streamlit server.

🔮 Future Improvements

[ ] Add prediction confidence scores (percentage).

[ ] Integrate probability bar visualizations.

[ ] Compare performance across multiple models (RoBERTa, ALBERT).

[ ] Expand the dataset for better cross-domain generalization.

[ ] Build a standalone REST API version.

👩‍💻 Author

Mano Rakshitha AI Engineering Student Building production-ready NLP systems and exploring the frontiers of Deep Learning.

<p align="center">
⭐️ If you find this project helpful, give it a star!
</p>


