import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import NotionDirectoryLoader
from qdrant_client.models import VectorParams, PointStruct

NO_OF_SIMILAR_CHUNKS = 4
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 10


def create_instructor_xl():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceInstructEmbeddings(model_name=EMB_INSTRUCTOR_XL, model_kwargs={"device": device})


def init_embedding(embedding_name):
    if embedding_name == EMB_INSTRUCTOR_XL:
        return create_instructor_xl()
    else:
        raise ValueError("Invalid config")


class MasterEmbeddings:
    """
    This class is responsible for creating vector DB and storing it in Qdrant Cloud
    """

    def __init__(self, config,model_name="bert-base-uncased"):
        self.config = config
        self.embeddings = None
        self.qdrant_client = None

    def init_embeddings(self):
        """
        Initialize the embeddings
        :return:
        """
        self.embeddings = init_embedding(self.config["embedding"])

    def init_qdrant_client(self):
        """
        Initialize the Qdrant client
        :return:
        """
        # Replace with your Qdrant Cloud API key and URL
        self.qdrant_client = QdrantClient(
            url=self.config["qdrant_url"], api_key=self.config["qdrant_api_key"]
        )

    def create_vector_db(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        # Load documents, split them into chunks, and generate embeddings
        documents = self._load_documents(self.config["docs_path"])
        vector_chunks = self._chunk_documents(documents, chunk_size, chunk_overlap)

        # Upload vectors to Qdrant
        self._upload_vectors_to_qdrant(vector_chunks)

    def _load_documents(self, docs_path):
        # Implement your document loading logic here
        return ["document1", "document2"]  # Placeholder example

    def _chunk_documents(self, documents, chunk_size, chunk_overlap):
    # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_texts = []
        for doc in documents:
            chunks = text_splitter.split_text(doc)  # Splitting each document into chunks of text
            chunked_texts.extend(chunks)

        # Embed each chunk of text using the embeddings model
        return [self.embeddings.embed_documents([chunk]) for chunk in chunked_texts]
        # Embed each chunk using the embeddings model
        return [self.embeddings.embed_documents([chunk]) for chunk in chunked_docs]

    def _upload_vectors_to_qdrant(self, vector_chunks):
        """
        Upload vector embeddings to Qdrant
        :param vector_chunks: List of vector embeddings
        :return:
        """
        collection_name = self.config["collection_name"]

        # Check if the collection exists, if not create it
        if collection_name not in [col.name for col in self.qdrant_client.get_collections().collections]:
            # Define the vector configuration
            vector_size = len(vector_chunks[0])  # Length of the first vector
            vectors_config = {
                "default": VectorParams(
                    size=vector_size,  # Specify the size of vectors (vector dimensionality)
                    distance="Cosine"  # Specify the distance metric
                )
            }

            # Create the collection with the vector configuration
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config  # Pass the vector configuration
            )

        # Upload the vector chunks as points to Qdrant
        points = [
            PointStruct(id=i, vector={"default": vector})  # Name the vector explicitly as "default"
            for i, vector in enumerate(vector_chunks)
        ]
        self.qdrant_client.upsert(collection_name=collection_name, points=points)


if __name__ == "__main__":
    EMB_INSTRUCTOR_XL = "hkunlp/instructor-xl"
    embedding_config = {
    "embedding": EMB_INSTRUCTOR_XL,
    "docs_path": {
        "product_db": "/Users/harish/Documents/varun(unemployed)/docs",
    },
    "qdrant_url": "https://91192707-948f-4785-b4f6-348c269428ab.europe-west3-0.gcp.cloud.qdrant.io",  # Replace with your Qdrant Cloud URL
    "qdrant_api_key": "4dJlEkfLKM0SDdO86qtwf_SMOHLiWHBTLammQRLnCz6ddMN5jasNNg",        # Replace with your Qdrant Cloud API key
    "collection_name": "PDF_COLLECTIONS1"       # Replace with the name of the collection to store embeddings
}

    config = embedding_config
    emb = MasterEmbeddings(config)
    emb.init_embeddings()
    emb.init_qdrant_client()  # Initialize the Qdrant client
    emb.create_vector_db()
