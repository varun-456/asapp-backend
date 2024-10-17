from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import pdfplumber
from typing import List
import cohere
from qdrant_client import models, QdrantClient
import uuid

# Initialize FastAPI app
app = FastAPI(
    title="PDF Chatbot API with Vector Storage",
    version="1.4",
    description="Chatbot with PDF upload, text extraction, and embeddings stored in Qdrant for vector search",
)

# Ensure Cohere API key is available
cohere_api_key = os.getenv("COHERE_API_KEY", "7niJGSzUgsOJFjtvdy6jOhvFK8rp0xnGdR9QVOAI")
if not cohere_api_key:
    raise ValueError("Cohere API key not set. Please set it using 'COHERE_API_KEY' environment variable.")

# Instantiate the Cohere client with the API key
cohere_client = cohere.Client(api_key=cohere_api_key)

# Qdrant setup
qdrant_url = "https://91192707-948f-4785-b4f6-348c269428ab.europe-west3-0.gcp.cloud.qdrant.io"
qdrant_api_key = os.getenv("QDRANT_API_KEY", "4dJlEkfLKM0SDdO86qtwf_SMOHLiWHBTLammQRLnCz6ddMN5jasNNg")
qdrant_client = QdrantClient(qdrant_url, prefer_grpc=False, api_key=qdrant_api_key)
QDRANT_COLLECTION_NAME = "PDF_DOCUMENTS"

# Define the model (using Cohere)
model = ChatCohere()

# Create the prompt template for context
system_template = "You are a helpful assistant. Current context: {context}"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# Define the chain (prompt template, model, and parser)
parser = StrOutputParser()
chain = prompt_template | model | parser

# Store chat, PDF content, and embeddings
pdf_content_store = {}
chat_history_store = {}

# Helper function to extract text from PDF using pdfplumber
def extract_text_from_pdf(file: UploadFile) -> str:
    try:
        with pdfplumber.open(file.file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing PDF")

# Function to generate embeddings for text
def generate_embeddings(text: str):
    # Ensure the correct model is used for 1024-dimension embeddings (check Cohere docs for exact model)
    response = cohere_client.embed(texts=[text], model="embed-english-v2.0")  # Use a 1024-dimension model
    return response.embeddings[0]

# Function to store embeddings in Qdrant
def store_embeddings_in_qdrant(embedding, user_id, document_name, metadata):
    vector = embedding
    
    # Generate a unique UUID for the document
    document_uuid = str(uuid.uuid4())
    
    # Payload contains the original file name and other metadata
    payload = {
        "user_id": user_id,
        "document_name": document_name,  # Store the original file name in the payload
        "metadata": metadata,
    }
    
    # Create point structure (ID, vector, and payload) for Qdrant
    point = models.PointStruct(id=document_uuid, vector=vector, payload=payload)
    
    # Use the correct upsert method to upload points (vectors) to Qdrant
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[point]
    )

# API Endpoint for uploading multiple PDFs and storing their content and embeddings
@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...), user_id: str = "default_user"):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    extracted_texts = []
    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")
        
        extracted_text = extract_text_from_pdf(file)
        extracted_texts.append(extracted_text)

    if user_id in pdf_content_store:
        pdf_content_store[user_id].extend(extracted_texts)
    else:
        pdf_content_store[user_id] = extracted_texts

    # Store extracted text and embeddings in Qdrant
    for i, extracted_text in enumerate(extracted_texts):
        embedding = generate_embeddings(extracted_text)
        store_embeddings_in_qdrant(
            embedding=embedding,
            user_id=user_id,
            document_name=files[i].filename,
            metadata={"document_index": i, "file_name": files[i].filename}
        )

    if user_id not in chat_history_store:
        chat_history_store[user_id] = [
            SystemMessage(content="You are an AI assistant that helps users understand their PDF content.")
        ]

    return {"message": f"{len(files)} PDFs uploaded successfully and text extracted."}

# Request body model for the chatbot interactions
class ChatRequest(BaseModel):
    user_id: str
    question: str
    document_index: int = None  # Optional, allows user to ask from a specific document

# API Endpoint for asking questions based on the uploaded PDFs
@app.post("/ask/")
async def ask_question(request: ChatRequest):
    user_id = request.user_id
    question = request.question
    document_index = request.document_index

    if user_id not in pdf_content_store or len(pdf_content_store[user_id]) == 0:
        raise HTTPException(status_code=400, detail="Please upload PDFs first.")

    # Retrieve the embeddings for the question
    question_embedding = generate_embeddings(question)

    # Perform a search in Qdrant to retrieve the most relevant document(s)
    search_results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=question_embedding,
        limit=1  # Adjust this to retrieve more documents if needed
    )

    if len(search_results) == 0:
        raise HTTPException(status_code=404, detail="No matching documents found.")

    best_match = search_results[0]
    best_match_text = best_match.payload['metadata']['file_name']
    matched_pdf_content = pdf_content_store[user_id][int(best_match.payload['metadata']['document_index'])]

    # Generate a response based on the matched content
    chat_history = chat_history_store[user_id]
    chat_history.append(HumanMessage(content=question))

    combined_messages = chat_history + [
        HumanMessage(content=f"Answer this question based on the following PDF content:\n\n{matched_pdf_content}")
    ]

    context = "\n".join([msg.content for msg in combined_messages])

    chain_input = {
        "context": context,
        "text": question,
    }

    response = chain.invoke(chain_input)
    chat_history.append(AIMessage(content=response))

    return {"answer": response}

# Running the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
