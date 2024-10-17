import uuid
import cohere
import qdrant_client
from qdrant_client.http import models
import pdfplumber
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from retrying import retry
import numpy as np
import hashlib
import json

# Initialize Cohere client
cohere_api_key = "7niJGSzUgsOJFjtvdy6jOhvFK8rp0xnGdR9QVOAI"  # Replace with your actual Cohere API key
cohere_client = cohere.Client(cohere_api_key)

# Initialize Qdrant Cloud client (update with your Qdrant Cloud URL and API key)
qdrant_client = qdrant_client.QdrantClient(
    url="https://91192707-948f-4785-b4f6-348c269428ab.europe-west3-0.gcp.cloud.qdrant.io",  # Replace with your Qdrant Cloud URL
    api_key="4dJlEkfLKM0SDdO86qtwf_SMOHLiWHBTLammQRLnCz6ddMN5jasNNg",  # Replace with your Qdrant Cloud API key
    timeout=60  # Increase timeout to handle larger operations
)


# Folder where PDFs are stored
collection_name = "research_papers"

# Create a collection in Qdrant Cloud (update vector size to 4096 to match the embeddings)
try:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=4096, distance=models.Distance.COSINE),
    )
except Exception as e:
    print("Collection already exists:", e)

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path: str) -> str:
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Function to chunk text using Langchain's RecursiveCharacterTextSplitter
def chunk_text(text: str) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Customizable chunk size
        chunk_overlap=100,  # Define overlap between chunks for better context
        length_function=len,
        separators=["\n\n", "\n", " ", ""],  # Try splitting at paragraphs, lines, words
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate text embeddings using Cohere's default model (which generates 4096-dimensional embeddings)
def generate_embeddings(texts: list) -> list:
    response = cohere_client.embed(texts=texts).embeddings
    return response

# Retry mechanism for upsert operation
@retry(stop_max_attempt_number=5, wait_fixed=2000)
def upsert_with_retry(points_batch):
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points_batch
    )

# Function to process a single PDF file, chunk it, and store embeddings in Qdrant Cloud with batching and retry
def process_pdf(file_path: str, batch_size=10):
    # Extract text from PDF
    text = extract_text_from_pdf(file_path)

    # Chunk the text into smaller pieces using Langchain's RecursiveCharacterTextSplitter
    chunks = chunk_text(text)

    # Generate embeddings for the chunks
    embeddings = generate_embeddings(chunks)

    # Store embeddings in Qdrant Cloud in batches
    points_batch = []
    for idx, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        point = models.PointStruct(
            id=str(uuid.uuid4()),  # Generate a unique UUID for each point
            vector=embedding,
            payload={"filename": os.path.basename(file_path), "chunk": chunk}
        )
        points_batch.append(point)

        # Upsert in batches
        if len(points_batch) >= batch_size:
            upsert_with_retry(points_batch)
            points_batch = []  # Reset batch after upsert

    # Upsert any remaining points in the last batch
    if points_batch:
        upsert_with_retry(points_batch)

    print(f"Processed and stored embeddings for {file_path}.")

# Function to generate embedding for a user query and search for similar chunks in Qdrant Cloud
def search_similar_chunks(user_query: str, top_k=5):
    
    # Generate embedding for the user query
    query_embedding = generate_embeddings([user_query])[0]

    # Search for top_k similar chunks in Qdrant Cloud
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )

    # Extract context from search results
    similar_chunks = [
        {
            "chunk": result.payload["chunk"],
            "filename": result.payload["filename"],
            "score": result.score
        }
        for result in search_result
    ]

    
    return similar_chunks

# Function to generate an answer using the user query and the retrieved similar chunks
def generate_answer_with_context(user_query: str, similar_chunks: list) -> str:
    
    # Combine the similar chunks into context text
    context_text = "\n\n".join([chunk["chunk"] for chunk in similar_chunks])

    # Create a prompt combining user query and the context text
    prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\n\nProvide a summarized answer:"

    # Generate answer using Cohere's generate endpoint
    response = cohere_client.generate(
        model='command-xlarge-nightly',  # Use appropriate model for text generation
        prompt=prompt,
        max_tokens=150,  # Limit the response length as needed for a summary
        temperature=0.5,  # Adjust temperature for more deterministic responses
        stop_sequences=["\n"]
    )

    answer = response.generations[0].text.strip()

    
    return answer

# Example usage
if __name__ == "__main__":
    # Process a PDF file and store its embeddings
    pdf_path = "/Users/harish/Documents/varun(unemployed)/docs/1.pdf"  # Replace with your PDF file path
    # process_pdf(pdf_path)

    # Get user query, convert to embedding, and find similar chunks
    user_query = "what is c4?"  # Replace with your query
    similar_chunks = search_similar_chunks(user_query)

    # Generate answer with context
    answer = generate_answer_with_context(user_query, similar_chunks)

    # Print the answer
    print("Generated Answer:")
    print(answer)
