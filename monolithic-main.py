# Cmd to run the code
# uvicorn monolithic-main:app --host 0.0.0.0 --port 50 --reload

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import requests, os, json, re
from typing import Union, List
from PyPDF2 import PdfReader
import html2text
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np
from dotenv import load_dotenv
from transformers import pipeline
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables from the .env file
load_dotenv("./")
# Initiate FastAPI app
app = FastAPI()

#############################################################################################################

# Agent Orchestrator
class QuizRequest(BaseModel):
    document_file: Union[UploadFile, None] = None
    document_url: Union[str, None] = None
    query: str = ""

@app.post("/generate-quiz/")
async def generate_quiz(data: QuizRequest):
    # Step 1D: Parse document to smaller text chunks
    doc_parser_response = await parse_document(
        file=data.document_file,
        url=data.document_url
    )
    if not doc_parser_response.get("text"):
        return {"error": "Failed to parse document"}
    else:
        print("1D: Document parsing successfull")

    # Step 2D: Create chunks from the parsed data
    doc_split_chunks_response = chunk_text_service(
        {
            "chunk_size": 300,
            "chunk_overlap": 50,
            "text": doc_parser_response.text
        }
    )
    print("2D: Document chunk spliting successfull")

    # Step 3D: Convert document text chunks to vector embeddings
    doc_create_embeddings_response = get_embeddings_list(doc_split_chunks_response)
    if not doc_create_embeddings_response.get("embeddings"):
        return {"error": "Failed to generate embeddings from document"}
    else:
        print("3D: Document create embeddings successfull")

    # Step 4D: Add the document vector embeddings to FAISS for storage
    doc_storage_response = add_embedding(doc_create_embeddings_response)
    if not doc_storage_response.get("message"):
        return {"error": "Failed to add embeddings of document to the vector"}
    else:
        print("4D: Document add embeddings successfull")
    print(doc_storage_response.get("message"))

    # Step 1Q: Get vector embeddings for the query text
    query_embeddings_response = get_embeddings({
        "text": data.query
    })
    if not query_embeddings_response.get("embedding"):
        return {"error": "Failed to generate embeddings from query"}
    else:
        print("1Q: Query create embeddings successfull")
    query_embedding_input = {
        "query_embedding": query_embeddings_response["embedding"],
        "top_k": 5
    }

    # Step 2Q: Retrieve the final query string (query_retrieval_response) 
    # from the vector embeddings similar to the query vector embeddings 
    # from the FAISS storage
    query_retrieval_response = query_embedding(query_embedding_input)
    if not query_retrieval_response.get("concatenated_content_result"):
        return {"error": "Failed to query stored embeddings of document using prompt"}
    else:
        print("2Q: Query document embeddings successfull")

    # Step 5D: Clear the document vector embeddings in FAISS storage and indices
    doc_clear_embeddings_response = clear_embeddings()
    if not doc_clear_embeddings_response.get("message"):
        return {"error": "Failed to clear embeddings storage and indices"}
    print("5D: Document clear embeddings successfull")
    print(doc_clear_embeddings_response.get("message"))

    # Step 3Q: Generate quiz questions based on the query relevant text
    questions_response = generate_questions(query_retrieval_response)
    if not questions_response.get("quiz"):
        return {"error": "Failed to retrieve questions from the query relevant text"}
    else:
        print("3Q: Quiz generated successfully")
    return questions_response

#############################################################################################################

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

#############################################################################################################

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

#############################################################################################################

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

#############################################################################################################

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

@app.post("/clear-embeddings/")
def clear_embeddings():
    """
    Clears all embeddings from the FAISS index and storage.
    """
    global embeddings_storage, index
    # Reset the FAISS index
    index.reset()  # This clears all vectors from the index
    # Clear the embeddings storage
    embeddings_storage.clear()
    return {"message": "All embeddings and storage have been cleared successfully."}

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

#############################################################################################################

# Generate Questions using APIs from https://www.arliai.com/

# ArlAI API URL and API Key (replace `your_api_key` with your actual key). Get API details from environment variables
ARLIAI_API_URL = os.getenv("ARLIAI_API_URL")
ARLIAI_API_MODEL = os.getenv("ARLIAI_API_MODEL")
ARLIAI_API_KEY = os.getenv("ARLIAI_API_KEY")

if not ARLIAI_API_URL or not ARLIAI_API_KEY or not ARLIAI_API_MODEL:
    raise ValueError("Missing ARLIAI_API_URL or ARLIAI_API_KEY or ARLIAI_API_MODEL in environment variables")

# Function to call the ArlAI API
def generate_mcqs(input_text):
    headers = {
        "Authorization": f"Bearer {ARLIAI_API_KEY}",
        "Content-Type": "application/json"
    }

    # Define the best prompt
    prompt = f"""
    You are an expert quiz creator. Generate 5 multiple-choice questions (MCQs) based on the following text:
    {input_text}
    Each question should have:
    - 4 options that is not easily differentiable.
    - The unique one and only correct answer out of the 4 options, given seprately.
    Format the response as a JSON array like this:
    [
        {{
            "question": "What is the objective of the High Catch?",
            "options": [
                "To catch a ball which is bouncing quickly",
                "To catch a ball which is dropping quickly",
                "To catch a ball which is flying high",
                "To catch a ball which is moving slowly"
            ],
            "correct_answer": "To catch a ball which is dropping quickly"
        }},
        ...
    ]
    """
    payload = json.dumps({
        "model": "Meta-Llama-3.1-8B-Instruct",
        "prompt": prompt,
        "max_tokens": 1024,
        "temperature": 0.4,  # Adjust for creativity
        "stream": False
    })

    # Make the request
    response = requests.post(ARLIAI_API_URL, headers=headers, data=payload)

    if response.status_code == 200:
        return arliai_extract_quiz(response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

def arliai_extract_quiz(response):
    """
    Extracts the quiz questions from the API response and formats them into the desired JSON structure.

    Args:
        response (dict): The ARLIAI API response containing the quiz data.

    Returns:
        list: A list of dictionaries containing the quiz questions, options, and correct answers.
    """
    try:
        # Extract the quiz choices from the response
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("No choices found in the response.")
        
        # Parse the text containing the quiz JSON
        raw_text = choices[0].get("text", "")
        # Extract the JSON-like part of the text
        start = raw_text.find("[")
        end = raw_text.rfind("]") + 1
        if start == -1 or end == -1:
            raise ValueError("Quiz data is not in the expected format.")
        
        quiz_data = raw_text[start:end]
        parsed_json_quiz = parse_quiz_data(quiz_data)
        return parsed_json_quiz
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error extracting quiz data: {e}")
        return []

def parse_quiz_data(raw_data):
    """
    Extract and parse the JSON content from a raw response containing extraneous text.

    Args:
        raw_data (str): The raw data containing the JSON quiz and additional text.

    Returns:
        list: The parsed quiz as a list of dictionaries.
    """
    try:
        # Use regex to extract the JSON content
        json_match = re.search(r"\[\s*{.*?}\s*]", raw_data, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON found in the provided data.")
        json_content = json_match.group(0)
        
        # Parse the extracted JSON content
        quiz = json.loads(json_content)
        return quiz
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return []
    except ValueError as e:
        print(f"Error: {e}")
        return []

# Define input data model
class QuizPrompt(BaseModel):
    concatenated_content_result: str
    detailed_results: list

# Endpoint to generate questions
@app.post("/generate-questions/")
def generate_questions(data: QuizPrompt):
    quiz_results = generate_mcqs(data.concatenated_content_result)
    return {"quiz": quiz_results}

#############################################################################################################

# # Generate Questions using Huggung Face Models

# HF_TOKEN= os.getenv("HF_TOKEN")

# # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Llama-3.2-1B"

# # Initialize the HuggingFace pipeline for Llama 3.1
# generator = pipeline(
#     "text-generation",
#     model=model_id,
#     # use_auth_token=HF_TOKEN,  # Pass your token here
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
