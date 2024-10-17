from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import pdfplumber
from typing import List

# FastAPI app setup
app = FastAPI(
    title="PDF Chatbot API",
    version="1.3",
    description="A chatbot that allows you to upload multiple PDFs and ask questions based on their content using LangChain and Cohere",
)

# 1. Define the model (using Cohere)
os.environ["COHERE_API_KEY"] = "7niJGSzUgsOJFjtvdy6jOhvFK8rp0xnGdR9QVOAI"
model = ChatCohere()

# 2. Create the prompt template for context
system_template = "You are a helpful assistant. Current context: {context}"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')  # Ensure {text} is expected
])

# 3. Define the chain (prompt template, model, and parser)
parser = StrOutputParser()
chain = prompt_template | model | parser

# Store chat and PDF content in memory (for demonstration)
pdf_content_store = {}
chat_history_store = {}

# Request body model for the chatbot interactions
class ChatRequest(BaseModel):
    user_id: str
    question: str
    document_index: int = None  # Optional, allows user to ask from a specific document

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

# API Endpoint for uploading multiple PDFs and storing their content
@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...), user_id: str = "default_user"):
    # Check if all files are PDFs
    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")

    # Extract text from each uploaded PDF
    extracted_texts = [extract_text_from_pdf(file) for file in files]

    # Store the extracted texts in the content store for the user
    if user_id in pdf_content_store:
        pdf_content_store[user_id].extend(extracted_texts)  # Append new PDFs to existing ones
    else:
        pdf_content_store[user_id] = extracted_texts  # Initialize with the new PDFs

    # Initialize chat history for the user if not already done
    if user_id not in chat_history_store:
        chat_history_store[user_id] = [
            SystemMessage(content="You are an AI assistant that helps users understand their PDF content.")
        ]

    return {"message": f"{len(files)} PDFs uploaded successfully and text extracted."}

# API Endpoint for asking questions based on the uploaded PDFs
@app.post("/ask/")
async def ask_question(request: ChatRequest):
    user_id = request.user_id
    question = request.question
    document_index = request.document_index

    # Check if the user has uploaded any PDFs
    if user_id not in pdf_content_store or len(pdf_content_store[user_id]) == 0:
        raise HTTPException(status_code=400, detail="Please upload PDFs first.")

    # Retrieve the extracted PDF content and chat history for the user
    pdf_contents = pdf_content_store[user_id]
    chat_history = chat_history_store[user_id]

    # Determine which document to base the answer on
    if document_index is not None:
        if document_index < 0 or document_index >= len(pdf_contents):
            raise HTTPException(status_code=400, detail=f"Invalid document index. You have {len(pdf_contents)} documents uploaded.")
        selected_pdf_content = pdf_contents[document_index]
    else:
        # Combine content from all PDFs if no index is specified
        selected_pdf_content = "\n\n".join(pdf_contents)

    # Add the new question to the chat history
    chat_history.append(HumanMessage(content=question))

    # Create the prompt based on the selected PDF content and the chat history
    combined_messages = chat_history + [
        HumanMessage(content=f"Answer this question based on the following PDF content:\n\n{selected_pdf_content}")
    ]

    context = "\n".join([msg.content for msg in combined_messages])

    # Prepare input for the chain
    chain_input = {
        "context": context,
        "text": question,
    }

    # Run the chain and get the response
    response = chain.invoke(chain_input)

    # Add the AI's response to the chat history
    chat_history.append(AIMessage(content=response))

    return {"answer": response}

# Running FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
