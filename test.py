import os
import concurrent.futures
from fastapi import FastAPI, UploadFile
import pdfplumber
import re
import cohere
from qdrant_client import QdrantClient, models
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Cohere API Key
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "7niJGSzUgsOJFjtvdy6jOhvFK8rp0xnGdR9QVOAI")
co = cohere.Client(COHERE_API_KEY)

# Qdrant Cloud setup with REST fallback
qdrant_api_key = os.getenv("QDRANT_API_KEY", "4dJlEkfLKM0SDdO86qtwf_SMOHLiWHBTLammQRLnCz6ddMN5jasNNgy")
qdrant_url = "https://91192707-948f-4785-b4f6-348c269428ab.europe-west3-0.gcp.cloud.qdrant.io"
qdrant = QdrantClient(qdrant_url, prefer_grpc=False, api_key=qdrant_api_key)

# Ensure Qdrant collection exists
collection_name = "research_papers"
try:
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=4096, distance=models.Distance.COSINE)
    )
    print(f"Collection {collection_name} created successfully.")
except Exception as e:
    print(f"Error creating collection: {str(e)}")

# Batch size and concurrency settings
BATCH_SIZE = 50
MAX_WORKERS = 5

# Step 1: Extract text from PDF
def extract_text_from_pdf(file):
    text = ''
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Step 2: Chunk text into manageable pieces
def chunk_text(text, max_length=300):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > max_length:
            yield ' '.join(current_chunk)
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length
    if current_chunk:
        yield ' '.join(current_chunk)

# Step 3: Embed text chunks using Cohere API
def embed_text_chunks(chunks):
    response = co.embed(texts=chunks)
    return response.embeddings

# Step 4: Store embeddings in Qdrant with batching and parallelism
def store_embeddings_in_batches(embeddings, chunks):
    total_points = len(embeddings)
    num_batches = (total_points + BATCH_SIZE - 1) // BATCH_SIZE  # Calculate number of batches

    def upload_batch(start_idx, end_idx):
        batch_embeddings = embeddings[start_idx:end_idx]
        batch_chunks = chunks[start_idx:end_idx]
        points = [
            models.PointStruct(id=start_idx + j, vector=batch_embeddings[j], payload={"text": batch_chunks[j]})
            for j in range(len(batch_embeddings))
        ]
        qdrant.upsert(collection_name=collection_name, points=points)

    # Run upload in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, total_points)
            futures.append(executor.submit(upload_batch, start_idx, end_idx))

        # Wait for all threads to finish
        concurrent.futures.wait(futures)

# Step 5: Search for relevant chunks based on query
def search_query(query):
    query_embedding = co.embed(texts=[query]).embeddings[0]
    
    search_result = qdrant.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=5  # Return top 5 closest results
    )
    
    return [hit.payload['text'] for hit in search_result]

# Step 6: Generate answer based on retrieved chunks using Cohere
def generate_answer(query, relevant_chunks):
    context = " ".join(relevant_chunks)
    
    # Use a valid Cohere model like "command-xlarge-nightly" for text generation
    response = co.generate(
        model="command-xlarge-nightly",  # Corrected model name
        prompt=f"Context: {context}\n\nQuestion: {query}\n\nAnswer:",
        max_tokens=100,
        temperature=0.5,
        stop_sequences=["\n"]
    )
    return response.generations[0].text.strip()

# API endpoint to upload PDFs in batch and process them
@app.post("/upload_pdfs/")
async def upload_pdfs(files: list[UploadFile]):
    for file in files:
        # Step 1: Extract text from each uploaded PDF
        pdf_text = extract_text_from_pdf(file.file)
        
        # Step 2: Chunk the text
        chunks = list(chunk_text(pdf_text))
        
        # Step 3: Embed the chunks
        embeddings = embed_text_chunks(chunks)
        
        # Step 4: Store embeddings in Qdrant with batching
        store_embeddings_in_batches(embeddings, chunks)
    
    return {"message": f"{len(files)} PDFs processed and stored successfully"}

# API endpoint to query the system
@app.post("/query/")
async def answer_query(query: str):
    # Step 5: Search for the most relevant chunks
    relevant_chunks = search_query(query)
    
    # Step 6: Generate an answer using Cohere LLM
    answer = generate_answer(query, relevant_chunks)
    
    return {"answer": answer}

# Automatically start the uvicorn server when the script is run
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
