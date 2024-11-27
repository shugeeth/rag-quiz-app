# Cmd to run the code
# uvicorn monolithic-main:app --host 0.0.0.0 --port 50 --reload

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import requests
from typing import Union, List
from PyPDF2 import PdfReader
import html2text
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np
from transformers import pipeline
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter

app = FastAPI()

# Agent Orchestrator

class QuizRequest(BaseModel):
    document_file: Union[UploadFile, None] = None
    document_url: Union[str, None] = None
    query: str = ""

@app.post("/generate-quiz/")
def generate_quiz(data: QuizRequest):
    # Step 1D: Parse document to smaller text chunks
    doc_parser_response = requests.post(
        "http://localhost:5101/parse-document/", 
        json={
            "file": data.document_file,
            "url": data.document_url
        }
    )
    if doc_parser_response.status_code != 200:
        return {"error": "Failed to parse document"}

    # Step 2D: Convert document text chunks to vector embeddings
    doc_embeddings_response = requests.post(
        "http://localhost:5102/get-embeddings/", 
        json={
            "text": doc_parser_response.text
        }
    )
    if doc_embeddings_response.status_code != 200:
        return {"error": "Failed to generate embeddings from document"}
    doc_embedding = doc_embeddings_response.json()["embedding"]

    # Step 3D: Add the document vector embeddings to FAISS for storage
    doc_retrieval_response = requests.post(
        "http://localhost:5103/add-embedding/", 
        json={
            "embedding": doc_embedding
        }
    )
    if doc_retrieval_response.status_code != 200:
        return {"error": "Failed to add embeddings of document to the vector"}

    # Step 1Q: Get vector embeddings for the query text
    query_embeddings_response = requests.post(
        "http://localhost:5102/get-embeddings/", 
        json={
            "text": data.query
        }
    )
    if query_embeddings_response.status_code != 200:
        return {"error": "Failed to generate embeddings from query"}
    query_embedding = query_embeddings_response.json()["embedding"]

    # Step 2Q: Retrieve the final query string (query_retrieval_response) 
    # from the vector embeddings similar to the query vector embeddings 
    # from the FAISS storage
    query_retrieval_response = requests.post(
        "http://localhost:5103/query-embedding/", 
        json={"query_embedding": query_embedding}
    )
    if query_retrieval_response.status_code != 200:
        return {"error": "Failed to retrieve embeddings for query embeddings"}
    query_relevant_text = query_retrieval_response.json()["results"]

    # Step 3Q: Generate quiz questions based on the query relevant text
    question_response = requests.post(
        "http://localhost:5104/generate-questions/", 
        json={
            "text": query_relevant_text
        }
    )
    if question_response.status_code != 200:
        return {"error": "Failed to retrieve questions from the query relevant text"}
    questions = question_response.json()["questions"]
    return {"quiz": questions}

# DOC PARSER

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "".join(page.extract_text() for page in reader.pages)

def extract_text_from_url(url):
    response = requests.get(url)
    return html2text.html2text(response.text)

@app.post("/parse-document/")
async def parse_document(file: UploadFile = None, url: str = None):
    if file:
        content = extract_text_from_pdf(file.file)
    elif url:
        content = extract_text_from_url(url)
    else:
        return {"error": "Provide a file or URL"}
    return {"text": content}

# Chunking

# Input schema
class ParsedTextData(BaseModel):
    chunk_size: int = 300  # Default chunk size
    chunk_overlap: int = 50  # Default overlap
    text: str
    

@app.post("/chunk-text/")
def chunk_text_service(data: ParsedTextData):
    """
    API endpoint to chunk text into smaller pieces.
    
    Args:
        data (ParsedTextData): Contains the parsed text, chunk size, and overlap.

    Returns:
        dict: A list of chunks ready for embedding.
    """
    # Initialize the LangChain text splitter
    text_splitter = CharacterTextSplitter(
        separator="\n",  # Split by lines
        chunk_size=data.chunk_size,
        chunk_overlap=data.chunk_overlap,
    )
    
    # Generate chunks
    chunks = text_splitter.split_text(data.text)
    
    # Return chunks as a list of dictionaries for consistency
    return {"chunks": [{"content": chunk} for chunk in chunks]}

# Get Embedding for chunks

embeddings_model = HuggingFaceEmbeddings(model_name="bert-base-nli-mean-tokens")

# Define input schema for chunk and chunk list
class Chunk(BaseModel):
    content: str

class ChunkList(BaseModel):
    chunks: List[Chunk]

class TextData(BaseModel):
    text: str

@app.post("/get-embeddings/")
def get_embeddings(data: TextData):
    vector = embeddings_model.embed_query(data.text)
    return {"embedding": vector}

@app.post("/get-embeddings-list/")
def get_embeddings_list(data: ChunkList):
    embeddings_with_content = []
    # Iterate through each chunk and generate embeddings
    for chunk in data.chunks:
        embedding_vector = embeddings_model.embed_query(chunk.content)
        embeddings_with_content.append({
            "embedding": embedding_vector,
            "content": chunk.content
        })
    print(f"{len(embeddings_with_content)} embeddings created successfully!")
    return {"embeddings": embeddings_with_content}

# Store(Add) and Retrieve(Query) embeddings

dimension = 768  # Set according to your embeddings size
index = faiss.IndexFlatL2(dimension)

class Embedding(BaseModel):
    embedding: list
    content: str 

class EmbeddingList(BaseModel):
    embeddings: List[Embedding]

class QueryData(BaseModel):
    query_embedding: list
    top_k: int = 5

embeddings_storage = []

@app.post("/add-embedding/")
def add_embedding(data: EmbeddingList):
    vectors = []
    for embedding in data.embeddings:
        # Accmulate embeddings/vector to be indexed later
        vectors.append(embedding.embedding)
        # Store the embedding and associated content in storage
        embeddings_storage.append({
            "embedding": embedding.embedding,
            "content": embedding.content
        })
    # Add all vectors to the FAISS index
    index.add(np.array(vectors, dtype="float32"))
    return {"message": f"{len(data.embeddings)} embeddings and their content added successfully"}

@app.post("/query-embedding/")
def query_embedding(data: QueryData):
    query_vector = np.array([data.query_embedding], dtype="float32")
    similarity_scores, indices = index.search(query_vector, data.top_k)
    # Retrieve content and its score associated with the top-k indices
    faiss_results = [
        {
            "content": embeddings_storage[i]["content"],
            "score": float(similarity_scores[0][idx])  # Score for the corresponding result
        }
        for idx, i in enumerate(indices[0])
    ]
    # Concatenate the content of the top-k faiss_results into a single string
    concatenated_content = " ".join(result["content"] for result in faiss_results)
    return {
        "concatenated_content_result": concatenated_content,
        "detailed_results": faiss_results  # Include detailed results for debugging or further use
    }

# Generate Questions

hf_token = "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"  # Paste your token here
# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_id = "meta-llama/Llama-3.2-1B"

# Initialize the HuggingFace pipeline for Llama 3.1
generator = pipeline(
    "text-generation",
    model=model_id,
    # use_auth_token=hf_token,  # Pass your token here
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",  # Auto-select the available device (CPU/GPU)
    max_new_tokens=100,  # Restrict the number of tokens generated
    truncation=True,
)

# Wrap the pipeline with HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=generator)

# Define input data model
class QuizPrompt(BaseModel):
    concatenated_content_result: str
    detailed_results: list

# Endpoint to generate questions
@app.post("/generate-questions/")
def generate_questions(data: QuizPrompt):
    prompt = (
        f"Generate a quiz with 5 multiple-choice questions, including 4 options per question, "
        f"on the topic: {data.concatenated_content_result}. "
        f"Provide the output in JSON format with correct answers marked."
    )
    # Invoke the LLM with the prompt
    response = llm.invoke(prompt)
    return {"questions": response}

#############################################################################################################

# # Generate Questions

# hf_token = ""  # Paste your token here
# # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Llama-3.2-1B"

# # Initialize the HuggingFace pipeline for Llama 3.1
# generator = pipeline(
#     "text-generation",
#     model=model_id,
#     # use_auth_token=hf_token,  # Pass your token here
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",  # Auto-select the available device (CPU/GPU)
#     max_new_tokens=100,  # Restrict the number of tokens generated
#     truncation=True,
# )

# # Wrap the pipeline with HuggingFacePipeline
# llm = HuggingFacePipeline(pipeline=generator)

# # Define input data model
# class QuizPrompt(BaseModel):
#     concatenated_content_result: str
#     detailed_results: list

# # Endpoint to generate questions
# @app.post("/generate-questions/")
# def generate_questions(data: QuizPrompt):
#     prompt = (
#         f"Generate a quiz with 5 multiple-choice questions, including 4 options per question, "
#         f"on the topic: {data.concatenated_content_result}. "
#         f"Provide the output in JSON format with correct answers marked."
#     )
#     # Invoke the LLM with the prompt
#     response = llm.invoke(prompt)
#     return {"questions": response}
