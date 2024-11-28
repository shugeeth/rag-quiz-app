# Quiz Generation Service

This project implements a **Quiz Generation Service** using **FastAPI**. It provides an end-to-end pipeline for parsing documents, generating embeddings, storing/retrieving embeddings using FAISS, and creating quiz questions based on user queries. The service supports input via file uploads or URLs.

---

## Features

1. **Document Parsing**:  
   - Extracts text from PDF files or webpage URLs.

2. **Text Chunking**:  
   - Splits long texts into manageable chunks with specified size and overlap for processing.

3. **Embeddings Generation**:  
   - Converts text chunks into vector embeddings using `HuggingFaceEmbeddings`with model `bert-base-nli-mean-tokens`.

4. **Embeddings Storage and Retrieval**:  
   - Stores embeddings in a FAISS index for similarity searches.  
   - Retrieves the most relevant chunks for a given query.

5. **Quiz Generation**:  
   - Uses the `ARLIAI API` with model `Meta-Llama-3.1-8B-Instruct` to generate multiple-choice quiz questions based on relevant document chunks.

6. **Embeddings Management**:  
   - Supports adding, querying, and clearing embeddings in FAISS.

---

## Requirements

- Python 3.8+
- Libraries:
  - `FastAPI`
  - `PyPDF2`
  - `html2text`
  - `faiss`
  - `langchain_huggingface`
  - `dotenv`
- FAISS for vector similarity search
- ARLIAI API for quiz generation (API key required)

---