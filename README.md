# 👁️ Local RAG Assistant

An offline, local-first Retrieval-Augmented Generation (RAG) pipeline designed for developers. It combines chunked official PDF documentation with atomic script ingestion to deliver repo-specific code snippets and technical guidance via Ollama (`llama3`).

---

## 📌 Architecture Overview

Unlike standard RAG implementations that chunk all documents uniformly, this system employs a **hybrid ingestion strategy**:

* **PDF Documentation**: Chunked semantically using `PyMuPDF` for broad context and operator parameter reference.
* **HALCON Code (`.hdev`)**: Custom XML-parsed and ingested as **atomic, un-chunked files** to preserve full script integrity, parameter logic, and execution context.
* **Vector Store**: Single-collection `ChromaDB` instance storing both structured manual context and complete script files.
* **Local Inference**: Runs locally via `Ollama` (`llama3`), utilizing high context limits (`num_ctx`: 16384, `num_predict`: 2048) to process whole files without context truncation or connection resets.

---

## 🛠️ Tech Stack

* **LLM & Embeddings:** Ollama (`llama3` model & embeddings)
* **Framework:** LangChain (`langchain-ollama`, `langchain-chroma`)
* **Vector Store:** ChromaDB
* **Document Loaders:** Custom `HDevLoader` (XML parsing), `PyMuPDFLoader`
* **Frontend:** Streamlit

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have [Ollama](https://ollama.com/) installed and running locally with the `llama3` model pulled:
```bash
ollama pull llama3
```
Install the required libraries using a pip.
```bash
pip install -r reqs.txt
```

### 2. Cold Start
To first initialize chroma_db, running the 'code_rag_cli.py' is necessary.
```bash
python code_rag_cli.py
```

### 3. Running the Web App
Now that you have initialized your chrome database and loaded your code and pdfs, you can run the web app using Streamlit.
```bash
streamlit run app.py
```

