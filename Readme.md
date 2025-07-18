# 🧠 Building Applications with Vector Databases

This project showcases six powerful application types built using vector databases. 

Have Fun! Reach out if you have any questions or need further assistance: https://himanibirole.com/
---

## 📚 Table of Contents

- [Overview](#overview)
- [Tech Stack & Dependencies](#tech-stack--dependencies)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Applying to Different Vector DBs](#applying-to-different-vector-dbs)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## 🧩 Overview

This repo provides hands-on examples of key vector-powered applications:

1. **Semantic Search** – Retrieve relevant text using embeddings  
2. **RAG (Retrieval-Augmented Generation)** – Contextualize LLM output using vector retrieval  
3. **Anomaly Detection** – Detect outliers via vector similarity  
4. **Hybrid Search** – Combine semantic and metadata filtering  
5. **Image Similarity Search** – Find visually similar images using embeddings  
6. **Recommender Systems** – Recommend items based on embedding proximity  


---

## ⚙️ Tech Stack & Dependencies

- Python 3.8+
- [LangChain](https://github.com/langchain-ai/langchain)
- [Pinecone](https://www.pinecone.io)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- Optional: FAISS, Chroma, Qdrant, Weaviate, pgvector
- Streamlit / Plotly for visualization

### 🔧 Requirements

##Install with:

```bash
pip install -r requirements.txt


##Core dependencies
langchain>=0.2.0
openai>=1.0.0
pinecone-client>=3.0.0
tqdm
pandas
numpy
scikit-learn
matplotlib

##Embeddings
sentence-transformers
transformers
torch

##Optional: Visualization/UI
streamlit
plotly

##Vector DB alternatives
faiss-cpu
chromadb
qdrant-client
weaviate-client
pgvector

##Notebook support
jupyterlab
ipykernel

##*Project Structure*
.
├── requirements.txt
├── notebooks/                  # Each notebook represents an application
│   ├── semantic_search.ipynb
│   ├── rag_app.ipynb
│   ├── anomaly_detection.ipynb
│   ├── hybrid_search.ipynb
│   ├── image_similarity.ipynb
│   └── recommender_system.ipynb
├── src/                        # Modular Python code for embedding and vector DB logic
│   ├── embeddings.py
│   └── vector_store.py
├── demos/                      # Optional UIs using Streamlit or FastAPI
├── data/                       # Sample or demo datasets
└── README.md


---

#🙏 Acknowledgments
This project is inspired by the Building Applications with Vector Databases course created by Tim Tully and presented by DeepLearning.AI, in partnership with Pinecone.

##Special thanks to:

1.DeepLearning.AI for educational content and inspiration
2.Pinecone for vector database infrastructure and tutorials
