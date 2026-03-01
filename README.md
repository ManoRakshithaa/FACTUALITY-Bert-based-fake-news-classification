# 🛡️ Factuality
### BERT-Powered Fake News Classification

[ **[Live Demo](https://factuality-bert-based-fake-news-classification.streamlit.app/)** ] · [ **[HuggingFace Model](https://huggingface.co/ManoRakshitha/factuality-distilbert)** ] · [ **[Dataset](https://www.kaggle.com/datasets/studymart/welfake-dataset-for-fake-news)** ]

---

**Factuality** is a deep learning application built to identify misinformation. By leveraging a fine-tuned **DistilBERT** transformer, it provides real-time news verification with the speed of a lightweight architecture and the precision of modern NLP.

---

## Project Overview
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

The system is designed with a decoupled architecture to ensure the repository remains lightweight while maintaining high performance.

* **Model Hosting:** Trained weights and configurations are hosted on the **HuggingFace Model Hub** to avoid large file storage in Git.
* **Frontend:** The interactive UI is deployed via **Streamlit Cloud**, which automatically syncs with this GitHub repository.
* **Inference:** When a user inputs text, the application fetches the model using the `transformers` API for real-time inference on the Streamlit server.

---

## Future Roadmap

I am actively working on expanding the capabilities of **Factuality**. The following features are prioritized for upcoming releases:

- [ ] **Prediction Confidence:** Display exact probability percentages for every classification.
- [ ] **Interactive Visuals:** Integrate probability bar charts to visualize model "certainty."
- [ ] **REST API:** Build a standalone API endpoint for third-party application integration.

---

### 👩‍💻 Author

**Mano Rakshitha** *AI Engineering Student*

Building production-ready NLP systems and exploring the frontiers of Deep Learning.

## Connect with me:
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mano-rakshitha) 
[![GitHub](https://img.shields.io/badge/GitHub-181717.svg?logo=github&logoColor=white)](https://github.com/ManoRakshithaa)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FFD21E?logoColor=black)](https://huggingface.co/ManoRakshitha)

