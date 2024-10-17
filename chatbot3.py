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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PDF Chatbot API with Vector Storage",
    version="1.5",
    description="Chatbot with PDF upload, text extraction, and embeddings stored in Qdrant for vector search",
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

# Helper function to extract text from PDF using pdfplumber
def extract_text_from_pdf(file: UploadFile) -> str:
    try:
        with pdfplumber.open(file.file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text.strip()  # Ensure no leading/trailing spaces
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail="Error processing PDF")

# Function to generate a single embedding for combined document text
def generate_embedding_for_combined_text(text: str) -> list:
    response = cohere_client.embed(texts=[text], model="embed-english-v2.0")  # Use a 1024-dimension model
    return response.embeddings[0]

# Function to store embeddings in Qdrant
def store_embeddings_in_qdrant(embedding, user_id, combined_text, doc_indices):
    document_uuid = str(uuid.uuid4())
    
    payload = {
        "user_id": user_id,
        "document_name": "Combined Document",
        "metadata": {
            "combined_text": combined_text,  # Store combined text
            "doc_indices": doc_indices  # Store indices of documents
        },
    }
    
    point = models.PointStruct(id=document_uuid, vector=embedding, payload=payload)
    
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[point]
    )

# API Endpoint for uploading multiple PDFs and storing their content and embeddings
@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...), user_id: str = "default_user"):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    combined_text = ""
    document_indices = []

    # Process each uploaded PDF
    for i, file in enumerate(files):
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")
        
        extracted_text = extract_text_from_pdf(file)
        combined_text += extracted_text + "\n\n"  # Combine extracted texts with separation
        document_indices.append(i)  # Keep track of document indices

    # Store the combined text for the user
    pdf_content_store.setdefault(user_id, []).append(combined_text)

    # Generate a single embedding for the combined text
    combined_embedding = generate_embedding_for_combined_text(combined_text)

    # Store the combined embedding in Qdrant with metadata
    store_embeddings_in_qdrant(
        embedding=combined_embedding,
        user_id=user_id,
        combined_text=combined_text,
        doc_indices=document_indices  # Store document indices for reference
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
    question_embedding = generate_embedding_for_combined_text(question)

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
            matched_contents.append(payload["metadata"]["combined_text"])  # Use the combined text

    if not matched_contents:
        raise HTTPException(status_code=404, detail="No matching documents found for the user.")

    # Combine contents for the response, limiting the number of characters for clarity
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
