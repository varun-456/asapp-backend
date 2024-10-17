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
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PDF Chatbot API with Vector Storage",
    version="2.0",
    description="Optimized Chatbot with PDF key section extraction, vector storage, and fast search.",
)

# Ensure Cohere API key is available
cohere_api_key = os.getenv("COHERE_API_KEY", "7niJGSzUgsOJFjtvdy6jOhvFK8rp0xnGdR9QVOAI")
if not cohere_api_key:
    raise ValueError("Cohere API key not set. Please set it using 'COHERE_API_KEY' environment variable.")

# Instantiate the Cohere client with the API key
cohere_client = cohere.Client(api_key=cohere_api_key)

# Qdrant setup
qdrant_url = "https://91192707-948f-4785-b4f6-348c269428ab.europe-west3-0.gcp.cloud.qdrant.io"  # Adjust your Qdrant URL
qdrant_api_key = os.getenv("QDRANT_API_KEY", "4dJlEkfLKM0SDdO86qtwf_SMOHLiWHBTLammQRLnCz6ddMN5jasNNg")
if not qdrant_api_key:
    raise ValueError("Qdrant API key not set. Please set it using 'QDRANT_API_KEY' environment variable.")
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

# Store chat and PDF content
pdf_content_store = {}
chat_history_store = {}

# Helper function to extract key sections from PDF using pdfplumber
def extract_key_sections_from_pdf(file: UploadFile) -> dict:
    try:
        with pdfplumber.open(file.file) as pdf:
            text = ""
            key_sections = {'title': '', 'abstract': '', 'conclusion': ''}
            
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
                
                # Simplified extraction logic (you may improve it further)
                if "abstract" in page_text.lower() and not key_sections['abstract']:
                    key_sections['abstract'] = page_text
                elif "conclusion" in page_text.lower() and not key_sections['conclusion']:
                    key_sections['conclusion'] = page_text
                # Extract title from first page
                if page.page_number == 1:
                    key_sections['title'] = page_text.split('\n')[0].strip()
            
            # Return only the essential sections for embedding
            return key_sections
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail="Error processing PDF")

# Function to generate embeddings for text
async def generate_embedding_for_text(text: str) -> list:
    response = await asyncio.to_thread(cohere_client.embed, texts=[text], model="embed-english-v2.0")  # Use a 1024-dimension model
    return response.embeddings[0]

# Function to store embeddings in Qdrant
def store_embeddings_in_qdrant(embedding, user_id, key_sections, doc_index):
    document_uuid = str(uuid.uuid4())
    
    payload = {
        "user_id": user_id,
        "document_name": f"Document {doc_index}",
        "metadata": {
            "title": key_sections['title'],
            "abstract": key_sections['abstract'],
            "conclusion": key_sections['conclusion'],
            "doc_index": doc_index  # Store the index of the document
        },
    }
    
    point = models.PointStruct(id=document_uuid, vector=embedding, payload=payload)
    
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[point]
    )

# API Endpoint for uploading multiple PDFs and storing their key sections in Qdrant
@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...), user_id: str = "default_user"):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    document_indices = []
    tasks = []  # List to hold embedding generation tasks

    # Process each uploaded PDF
    for i, file in enumerate(files):
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")
        
        key_sections = extract_key_sections_from_pdf(file)
        # Combine the key sections into a single text for embedding
        combined_text = f"Title: {key_sections['title']}\nAbstract: {key_sections['abstract']}\nConclusion: {key_sections['conclusion']}"
        
        # Generate embedding for the key sections
        tasks.append(generate_embedding_for_text(combined_text))
        document_indices.append((i, key_sections))  # Track doc index and key sections

    # Generate embeddings for all documents concurrently
    embeddings = await asyncio.gather(*tasks)

    # Store each embedding in Qdrant with metadata
    for embedding, (index, key_sections) in zip(embeddings, document_indices):
        store_embeddings_in_qdrant(
            embedding=embedding,
            user_id=user_id,
            key_sections=key_sections,  # Store the key sections
            doc_index=index  # Store the document index for reference
        )

    chat_history_store.setdefault(user_id, [
        SystemMessage(content="You are an AI assistant that helps users understand their PDF content.")
    ])

    logger.info(f"{len(files)} PDFs uploaded successfully for user {user_id}.")
    return {"message": f"{len(files)} PDFs uploaded successfully and text extracted."}

# Request body model for chatbot interactions
class ChatRequest(BaseModel):
    user_id: str
    question: str

# API Endpoint for asking questions based on the uploaded PDFs
@app.post("/ask/")
async def ask_question(request: ChatRequest):
    user_id = request.user_id
    question = request.question

    if user_id not in pdf_content_store or len(pdf_content_store[user_id]) == 0:
        raise HTTPException(status_code=400, detail="Please upload PDFs first.")

    # Retrieve the embeddings for the question
    question_embedding = await generate_embedding_for_text(question)

    # Perform a search in Qdrant to retrieve relevant document(s)
    search_results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=question_embedding,
        limit=10  # Retrieve up to 10 documents (adjust as necessary)
    )

    if len(search_results) == 0:
        raise HTTPException(status_code=404, detail="No matching documents found.")

    # Prepare content from the best matches based on user_id
    matched_contents = []
    for result in search_results:
        payload = result.payload
        if payload["user_id"] == user_id:  # Ensure the document belongs to the user
            matched_contents.append(f"Title: {payload['metadata']['title']}\nAbstract: {payload['metadata']['abstract']}\nConclusion: {payload['metadata']['conclusion']}")

    if not matched_contents:
        raise HTTPException(status_code=404, detail="No matching documents found for the user.")

    # Combine contents for the response
    combined_contents = "\n\n".join(matched_contents)

    # Generate a response based on the matched content
    chat_history = chat_history_store[user_id]
    chat_history.append(HumanMessage(content=question))

    combined_message = HumanMessage(content=f"Answer this question based on the following PDF contents:\n\n{combined_contents}")
    chat_history.append(combined_message)

    context = "\n".join([msg.content for msg in chat_history])

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
