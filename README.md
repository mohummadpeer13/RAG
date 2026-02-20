# RAG Scanner Toolkit

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey)
![Ollama](https://img.shields.io/badge/Powered%20by-Ollama-black)
![Status](https://img.shields.io/badge/Status-Active-success)

> Lightweight local toolkit for scanning documents, generating embeddings, managing a RAG index, and visualizing chunks & statistics.

---

## 📌 Overview

**RAG Scanner Toolkit** is a local-first utility designed to:

- 📂 Scan and process document collections  
- 🧠 Generate embeddings using Ollama  
- 🗑 Create and delete a RAG vector index  
- 🔍 Visualize generated chunks  
- 📊 Display dataset statistics  

Perfect for:

- RAG experimentation  
- Chunking strategy testing  
- Embedding inspection  
- Local AI prototyping  

---

## 🏗 Architecture

Pipeline workflow:

1. Document scanning  
2. Text extraction  
3. Chunking  
4. Embedding generation (Ollama)  
5. Vector indexing  
6. Statistics & visualization  

Models used:

- `llama3.1:8b` → LLM  
- `nomic-embed-text` → Embedding model  

---

## ⚙️ Requirements

- Linux (Ubuntu/Debian recommended)  
- Python 3.10+  
- Ollama installed  
- 16GB+ RAM recommended  

---

## 🚀 Ollama Installation on Machine

### 1️⃣ Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2️⃣ Pull Required Models

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Verify installation:

```bash
ollama list
```

### 3️⃣ Check Ollama Base URL

By default, Ollama runs a local API server at:

```
http://127.0.0.1:11434
```

---

## 🚀 Project Installation

You can run the RAG Scanner Toolkit either:

- 🖥 Locally (recommended for development)
- 🐳 With Docker (isolated & portable)

---

## 🖥 Local Installation
```bash
sudo apt update
sudo apt install python3-full python3-venv -y

python3 -m venv rag_env
source rag_env/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py

Open in browser:
http://127.0.0.1:8000/
```
---

## 🐳 Docker Installation

🐳 Build Image

```bash
docker compose up --build

```

🔧 Start Services

```bash
http://127.0.0.1:8000/
```

🧹 Stop Containers

```bash
docker compose down
```

🗑 Remove volumes:

```bash
docker compose down -v
```

📦 Data Persistence

- `data/` → source documents persist  
- `chroma_db/` → vector index persists  

---

The application enables you to:

- Scan a directory of documents  
- Build or delete the RAG index  
- Inspect generated chunks  
- View processing statistics  

---

## 📂 Project Structure

```
.
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/             # Source documents
├── templates/        # Views (HTML)
├── chroma_db/        # Vector index (Chroma persistence)
└── rag_env/          # Virtual environment (local only, not used in Docker)
```

---

## 🔍 Features

### 📂 Document Scanning
- Recursive directory parsing  
- Text extraction  
- Automatic chunking  

### 🧠 Embedding Generation
- Uses `nomic-embed-text` via Ollama  
- Local storage of embeddings  
- Fully offline  

### 🗑 RAG Index Management
- Create vector index  
- Safe deletion  
- Full rebuild support  

### 🔎 Chunk Visualization
- Inspect chunk content  
- View chunk sizes  
- Track source mapping  

### 📊 Statistics Dashboard
- Total documents processed  
- Total chunks generated  
- Average chunk size  
- Distribution per file  

---

## 🔐 Local-First Design

- No cloud dependency  
- No external API calls  
- 100% offline RAG experimentation  
- All data remains on your machine  

---

## 🛣 Roadmap

- Web UI  
- Configurable chunk size & overlap  
- FAISS / Chroma integration  
- Query interface with LLM response generation  
- Docker support  

---

## 📜 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository  
2. Create a feature branch  
3. Commit your changes  
4. Open a Pull Request  
