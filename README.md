# 🛡️ Factuality
### BERT-Powered Fake News Classification

[ **[Live Demo](https://factuality-bert-based-fake-news-classification.streamlit.app/)** ] · [ **[HuggingFace Model](https://huggingface.co/ManoRakshitha/factuality-distilbert)** ] · [ **[Dataset](https://www.kaggle.com/datasets/studymart/welfake-dataset-for-fake-news)** ] · [ **[Docker Hub](https://hub.docker.com/r/manorakshitha/factuality-app)** ]

---

**Factuality** is a deep learning–based fake news detection system built using DistilBERT, a lightweight transformer model from Hugging Face. The model was fine-tuned on the WELFake dataset (Kaggle) and evaluated on the ISOT Fake News Dataset to test cross-dataset generalization and real-world robustness.

The project focuses on building a high-accuracy binary classifier that distinguishes between real and fake news articles using contextual language understanding rather than simple keyword matching. The model achieved **~85–86% accuracy** during evaluation.

The trained model is deployed via Streamlit and containerized with Docker, allowing anyone to run the app locally in a single command — no setup required.

---

## Project Overview

This project demonstrates a full-cycle NLP workflow — from preprocessing raw Kaggle datasets to deploying a fine-tuned transformer model for public inference.

- **Objective:** Classify news articles as **Real** or **Fake**
- **Core Model:** DistilBERT (`distilbert-base-uncased`) for a balance of speed and accuracy
- **Deployment:** Decoupled architecture — HuggingFace Hub (model weights) + Streamlit Cloud (UI) + Docker (containerized local deployment)

---

## Tech Stack

| Category | Technology |
| :--- | :--- |
| **Language** | Python 3.10 |
| **ML Framework** | PyTorch, HuggingFace Transformers |
| **Architecture** | DistilBERT (`distilbert-base-uncased`) |
| **Web UI** | Streamlit |
| **Model Hosting** | HuggingFace Model Hub |
| **Containerization** | Docker |

---

## 🐳 Docker Deployment

Run the app locally in one command — no Python setup, no dependency issues:

```bash
docker pull manorakshitha/factuality-app:v1
docker run -p 8501:8501 manorakshitha/factuality-app:v1
```

Then open **http://localhost:8501** in your browser.

The container fetches the model from HuggingFace at runtime, so no weights are bundled in the image.

---

## Dataset & Pipeline

The model was trained on a comprehensive fake news dataset from Kaggle.

### Preprocessing Workflow

1. **Text Cleaning** — Stripping noise and special characters
2. **Tokenization** — Using the DistilBERT tokenizer
3. **Standardization** — Padding and truncation to `max_length` of 256
4. **Training** — Fine-tuned with `CrossEntropyLoss` and `AdamW` optimizer on a GPU-enabled environment

> [!IMPORTANT]
> Dataset files are not included in this repository due to licensing restrictions.

---

## 📂 Project Structure

Model weights are fetched dynamically from HuggingFace at runtime to keep the repository lightweight.

```
FACTUALITY-Bert-based-fake-news-classification/
├── 📄 app.py               # Streamlit frontend & inference logic
├── 📄 train.py             # Model fine-tuning script
├── 📄 requirements.txt     # Project dependencies
├── 🐳 Dockerfile           # Container build instructions
├── 📄 .dockerignore        # Docker build exclusions
└── 📄 README.md            # Documentation
```

---

## Deployment Architecture

The system uses a decoupled architecture to keep the repository lightweight while maintaining high performance.

- **Model Hosting** — Trained weights and configurations are stored on the HuggingFace Model Hub, avoiding large file storage in Git
- **Frontend** — The interactive UI is deployed on Streamlit Cloud, automatically synced with this GitHub repository
- **Containerization** — The full app is Dockerized for reproducible local deployment; anyone can pull and run it without configuring a Python environment
- **Inference** — When a user inputs text, the app fetches the model via the `transformers` API for real-time classification

---

## Future Roadmap

- [ ] **Prediction Confidence** — Display exact probability percentages for each classification
- [ ] **Interactive Visuals** — Integrate probability bar charts to visualize model certainty
- [ ] **REST API** — Build a standalone API endpoint for third-party integration

---

## 👩‍💻 Author

**Mano Rakshitha**
*AI Engineering Student*

Building production-ready ML systems and exploring the frontiers of Deep Learning.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mano-rakshitha)
[![GitHub](https://img.shields.io/badge/GitHub-181717.svg?logo=github&logoColor=white)](https://github.com/ManoRakshithaa)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FFD21E?logoColor=black)](https://huggingface.co/ManoRakshitha)
