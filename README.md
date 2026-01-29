# 🚀 code213 – Deep Learning & NLP Bootcamp  
**From Neurons to AI Agents (2026 AI Stack)**

Welcome to the **advanced stage** of the Data Science Bootcamp.  
This program traces the **evolution of Artificial Intelligence**, starting from first-principles implementations and progressing toward **production-grade AI systems** used in 2026 — including **YOLOv8, Transformers, Retrieval-Augmented Generation (RAG), and Agentic Workflows**.

> *The goal is not only to train models, but to engineer intelligent systems.*

---

## 🧭 Module Journey

### 1️⃣ Foundations — *From Scratch*
Before relying on frameworks, we master the mathematical and algorithmic foundations.

| Topic | Description |
|-----|------------|
| **Multi-Layer Perceptron (MLP)** | Implementing forward pass, backpropagation, and gradient descent using **NumPy only** |
| **Optimization Mathematics** | Constrained optimization with **Lagrange multipliers** for regularization |
| **Financial Intelligence** | ANN-based **fraud detection** for high-frequency transaction streams |

---

### 2️⃣ Computer Vision — *The Spatial Revolution*
Understanding how machines learn to see and reason spatially.

| Area | Content |
|----|--------|
| **CNN Emergence** | Why convolutions dominate spatial data |
| **Hall-of-Fame Architectures** | AlexNet, VGG, ResNet (residual learning) |
| **YOLOv8 Mastery** | Real-time object detection & instance segmentation |
| **Vision Transformers (ViT)** | Attention applied to image patches |
| **OpenCV Integration** | Real-time inference on camera streams |

---

### 3️⃣ NLP — *The Road to ChatGPT*
From sequence models to attention-based intelligence.

| Topic | Description |
|-----|------------|
| **Recurrent Architectures** | RNN, Bi-RNN, LSTM, GRU |
| **Transformer Architecture** | Implementing self-attention from scratch |
| **LLM Fine-Tuning** | Task-specific training (sentiment, classification, QA) |

---

### 4️⃣ AI Engineering & Agentic Workflows (2026 Stack)
Moving from models to **autonomous AI systems**.

| Technology | Purpose |
|---------|---------|
| **Vector Databases** | Semantic search with Pinecone / ChromaDB / FAISS |
| **RAG (Retrieval-Augmented Generation)** | Grounding LLMs in private & structured data |
| **Agentic AI** | Autonomous agents using LangChain & Claude 3.5 |
| **MCP (Model Context Protocol)** | FastMCP servers connecting LLMs to MySQL |

---

## 🛠️ Key Capstone Projects

| Project | Description |
|------|------------|
| 🏥 **Brain Tumor Detection & Case Retrieval System** | MRI tumor detection using CNNs **augmented with semantic search** to retrieve similar clinical cases |
| 💳 **Fraud Detection System** | Real-time ANN-based anomaly detection for financial transactions |
| 🍴 **Restaurant Recommendation Chatbot** | RAG-powered chatbot using **semantic search + sentiment analysis** |
| 📄 **PDF Question-Answering RAG System** | Secure document-grounded Q&A over uploaded PDFs |
| 📦 **Smart Inventory Manager** | Agentic AI system using FastMCP & SQL for warehouse automation |

---

## 🏥 Brain Tumor Detection & Semantic Case Search

**Objective**  
Go beyond classification by enabling **case-based reasoning**, similar to real clinical workflows.

**System Capabilities**
- CNN-based MRI tumor detection and classification
- Feature embedding extraction from intermediate layers
- Semantic similarity search over historical MRI cases
- Retrieval of:
  - Similar MRI scans
  - Diagnostic notes
  - Treatment summaries (when available)

**Why This Matters**
> Radiologists reason by comparison, not only prediction.

This system improves:
- Diagnostic confidence
- Explainability
- Medical education

**Tech Stack**
- `PyTorch` / `TensorFlow`
- Transfer Learning (ResNet, EfficientNet)
- `FAISS` / `ChromaDB`
- `Sentence-Transformers`

---

## 🍴 Restaurant Recommendation Chatbot  
### Semantic Search + Sentiment Analysis

**Objective**  
Build an emotionally aware AI assistant that recommends food intelligently.

**Core Intelligence**
- Semantic search over menu items (ingredients, cuisine, nutrition)
- Sentiment analysis of user input (tired, happy, stressed, excited)
- Context-aware ranking based on:
  - Mood
  - Dietary preferences
  - Time of day

**Example**
> *“I had a long day and want something comforting.”*  
→ Warm, high-satisfaction dishes are prioritized.

**Tech Stack**
- `Transformers` (sentiment models)
- `Sentence-Transformers`
- `LangChain`
- `ChromaDB` / `Pinecone`

---

## 📄 PDF Question-Answering RAG System

**Objective**  
Answer questions **strictly grounded** in uploaded documents, preventing hallucinations.

**Pipeline**
1. PDF ingestion & chunking  
2. Embedding generation  
3. Vector database indexing  
4. Retrieval of relevant chunks  
5. Context-aware answer generation (with sources)

**Use Cases**
- Research papers
- Medical reports
- Legal documents
- Internal company knowledge bases

**Tech Stack**
- `LangChain`
- `FAISS` / `ChromaDB`
- `Transformers`
- `PyPDF`

---

## ⚙️ Requirements & Setup

### 🔧 Prerequisites
- GPU-enabled environment (**CUDA recommended**)
- Python ≥ 3.10

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/dl-nlp-bootcamp.git
cd dl-nlp-bootcamp

# Install dependencies
pip install -r requirements.txt




