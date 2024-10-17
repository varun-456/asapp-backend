import os
import pdfplumber
import cohere
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams

# Initialize Cohere API (Gemini) and Qdrant Client
cohere_api_key = '7niJGSzUgsOJFjtvdy6jOhvFK8rp0xnGdR9QVOAI'  # Replace with your Cohere API key
qdrant_host = 'https://91192707-948f-4785-b4f6-348c269428ab.europe-west3-0.gcp.cloud.qdrant.io'  # Replace with your Qdrant Cloud host URL
qdrant_api_key = '4dJlEkfLKM0SDdO86qtwf_SMOHLiWHBTLammQRLnCz6ddMN5jasNNg'  # Replace with your Qdrant Cloud API key
collection_name = 'research_papers'

cohere_client = cohere.Client(cohere_api_key)
qdrant_client = QdrantClient(
    url=qdrant_host,
    api_key=qdrant_api_key
)

# Check if the collection exists in Qdrant, if not, create it
def create_collection_if_not_exists():
    collections = qdrant_client.get_collections().collections
    if not any(collection.name == collection_name for collection in collections):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=4096, distance="Cosine")
        )
        print(f"Collection '{collection_name}' created.")
    else:
        print(f"Collection '{collection_name}' already exists.")

# Extract text from a single PDF file
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Generate embeddings using Cohere API
def generate_embeddings(text):
    response = cohere_client.embed(texts=[text])
    return response.embeddings[0]

# Store embeddings in Qdrant Cloud
def store_embedding_in_qdrant(embedding, paper_id, pdf_name):
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[{
            "id": paper_id,
            "vector": embedding,
            "payload": {"file_name": pdf_name}
        }]
    )
    print(f"Stored embeddings for: {pdf_name}")

# Main function to process all PDFs in a folder
def process_pdfs_in_folder(folder_path):
    create_collection_if_not_exists()
    
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    for index, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"Processing {pdf_file}...")
        
        # Extract text from the PDF
        text = extract_text_from_pdf(pdf_path)
        
        if text:
            # Generate embeddings for the text
            embedding = generate_embeddings(text)
            
            # Store the embedding in Qdrant
            store_embedding_in_qdrant(embedding, paper_id=index, pdf_name=pdf_file)
        else:
            print(f"Could not extract text from: {pdf_file}")

# Set your folder path where the research papers are stored
folder_path = '/Users/harish/Documents/varun(unemployed)/docs'  # Replace with the actual folder path

# Run the PDF processing
process_pdfs_in_folder(folder_path)
