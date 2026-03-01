# 🛡️ Factuality: BERT-Based Fake News Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://factuality-bert-based-fake-news-classification.streamlit.app/)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-orange)](https://huggingface.co/ManoRakshitha/factuality-distilbert)

A real-time deep learning solution designed to combat misinformation. **Factuality** leverages a fine-tuned DistilBERT transformer to classify news text with high precision.


---

## 🧠 Project Overview
This project demonstrates a full-cycle NLP workflow—from preprocessing raw Kaggle datasets to deploying a fine-tuned transformer model for public inference.

* **Objective:** Classify news articles as ✅ **Real** or ❌ **Fake**.
* **Core Tech:** DistilBERT (`distilbert-base-uncased`) for a balance of speed and accuracy.
* **Deployment:** Seamless integration between HuggingFace Hub (Model) and Streamlit Cloud (UI).

## Tech Stack

| Category | Technology |
| :--- | :--- |
| **Language** | Python 3.10 |
| **ML Framework** | PyTorch, HuggingFace Transformers |
| **Architecture** | DistilBERT |
| **Web UI** | Streamlit |
| **Model Hosting** | HuggingFace Model Hub |

---

## Dataset & Pipeline
The model was trained on a comprehensive Fake News dataset from Kaggle. 

### Preprocessing Workflow:
1.  **Text Cleaning:** Stripping noise and special characters.
2.  **Tokenization:** Utilizing the specialized DistilBERT tokenizer.
3.  **Standardization:** Padding and truncation to a `max_length` of 256.
4.  **Training:** Fine-tuned using `CrossEntropyLoss` on a GPU-enabled environment via `AdamW` optimizer.

> [!IMPORTANT]
> Dataset files are not included in this repository due to licensing restrictions.

---

## 📂 Project Structure

To keep the repository lightweight, model weights are fetched dynamically from the **HuggingFace Hub** at runtime.

```bash
FACTUALITY-Bert-based-fake-news-classification
├── 📄 app.py              # Streamlit frontend & inference logic
├── 📄 requirements.txt    # Project dependencies
└── 📄 README.md           # Documentation
```

## Deployment Architecture

Model Hosting: Weights reside on HuggingFace Model Hub.
Frontend: Streamlit Cloud pulls the code from GitHub.
Inference: On-the-fly inference using the Transformers API on the Streamlit server.

## Future Roadmap
- Add prediction confidence scores (%).

- Integrate probability bar visualizations.

- Cross-model benchmarking (RoBERTa vs. ALBERT).

- Build a standalone REST API.

## Author
Mano Rakshitha
AI Engineering Student
www.linkedin.com/in/mano-rakshitha
