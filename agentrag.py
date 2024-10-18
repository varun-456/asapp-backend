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
import random

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
questions_collection_name = "user_questions"

# Create a collection in Qdrant Cloud (update vector size to 4096 to match the embeddings)
try:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=4096, distance=models.Distance.COSINE),
    )
except Exception as e:
    print("Collection already exists:", e)

# Create a collection for storing user questions
try:
    qdrant_client.create_collection(
        collection_name=questions_collection_name,
        vectors_config=models.VectorParams(size=4096, distance=models.Distance.COSINE),
    )
except Exception as e:
    print("Questions collection already exists:", e)

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
def upsert_with_retry(points_batch, collection_name):
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
            upsert_with_retry(points_batch, collection_name)
            points_batch = []  # Reset batch after upsert

    # Upsert any remaining points in the last batch
    if points_batch:
        upsert_with_retry(points_batch, collection_name)

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
def generate_answer_with_context(user_query: str, similar_chunks: list, temperature=0.5, max_tokens=150) -> dict:
    # Combine the similar chunks into context text
    context_text = "\n\n".join([chunk["chunk"] for chunk in similar_chunks])

    # Create a prompt combining user query and the context text
    prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\n\nProvide a summarized answer:"

    # Generate answer using Cohere's generate endpoint
    response = cohere_client.generate(
        model='command-xlarge-nightly',  # Use appropriate model for text generation
        prompt=prompt,
        max_tokens=max_tokens,  # Limit the response length as needed for a summary
        temperature=temperature,  # Adjust temperature for more deterministic or creative responses
        stop_sequences=["\n"]
    )

    answer = response.generations[0].text.strip()

    # Extract the filenames from the similar chunks
    filenames = list(set([chunk["filename"] for chunk in similar_chunks]))

    # Recommend potential next questions
    recommended_questions = recommend_next_questions(answer, top_k=3)

    return {
        "answer": answer,
        "filenames": filenames,
        "recommended_questions": recommended_questions
    }

# Function to store user questions in Qdrant Cloud
def store_user_question(question: str):
    # Generate embedding for the question
    question_embedding = generate_embeddings([question])[0]

    # Create a point with the question and its embedding
    point = models.PointStruct(
        id=str(uuid.uuid4()),  # Generate a unique UUID for each point
        vector=question_embedding,
        payload={"question": question}
    )

    # Upsert the question into the user questions collection
    upsert_with_retry([point], questions_collection_name)
    print(f"Stored user question: {question}")

# Function to recommend potential next questions to the user based on the generated answer
def recommend_next_questions(answer: str, top_k=3) -> list:
    # Use the generated answer to create follow-up questions
    prompt = f"Based on the following information, generate {top_k + 2} follow-up questions that a user might ask:\n\nAnswer: {answer}\n\nFollow-up Questions:"

    response = cohere_client.generate(
        model='command-xlarge-nightly',  # Use appropriate model for text generation
        prompt=prompt,
        max_tokens=200,  # Increase token limit for more comprehensive questions
        temperature=0.7,  # Adjust temperature for creative questions
        stop_sequences=["\n"]
    )

    follow_up_questions = response.generations[0].text.strip().split('\n')
    # Ensure there are at least three questions by initially generating more
    return follow_up_questions[:top_k] if len(follow_up_questions) >= top_k else follow_up_questions

# Function to handle feedback and retry if the user is not satisfied
def get_user_feedback_and_retry(user_query: str, similar_chunks: list):
    # Generate initial answer and recommendations
    result = generate_answer_with_context(user_query, similar_chunks)
    print("Generated Answer:")
    print(result["answer"])
    print("\nFiles Referenced:")
    print(result["filenames"])
    print("\nRecommended Questions:")
    for question in result["recommended_questions"]:
        print(question)

    # Ask for user feedback
    feedback = input("\nAre you satisfied with the answer? (yes/no): ")
    if feedback.lower() == 'no':
        # If not satisfied, retry with a different approach
        print("Retrying with a modified approach...\n")
        new_temperature = random.uniform(0.6, 0.9)  # Change temperature for more creative variation
        new_max_tokens = random.choice([150, 200, 250])  # Randomize max tokens to potentially provide more detail
        alternative_chunks = similar_chunks[::-1]  # Use a different order of chunks to change context

        result = generate_answer_with_context(user_query, alternative_chunks, temperature=new_temperature, max_tokens=new_max_tokens)
        print("Generated Answer (Retry):")
        print(result["answer"])
        print("\nFiles Referenced:")
        print(result["filenames"])
        print("\nRecommended Questions:")
        for question in result["recommended_questions"]:
            print(question)

# Example usage
if __name__ == "__main__":
    # Process a PDF file and store its embeddings
    pdf_path = "/Users/harish/Documents/varun(unemployed)/docs/1.pdf"  # Replace with your PDF file path
    # process_pdf(pdf_path)

    # Get user query, convert to embedding, and find similar chunks
    user_query = "what is c4?"  # Replace with your query
    similar_chunks = search_similar_chunks(user_query)

    # Get user feedback and retry if necessary
    get_user_feedback_and_retry(user_query, similar_chunks)

    # Store user question in Qdrant
    store_user_question(user_query)
