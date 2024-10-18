# Project: Retrieval-Augmented Generation (RAG) System

This project implements a Retrieval-Augmented Generation (RAG) system, allowing for efficient information retrieval and response generation using machine learning embeddings and vector storage. The system utilizes various components for chunking, embedding, vector storage, and lookups, orchestrated to provide seamless interaction with third-party large language models (LLMs).

## Project Structure

The project consists of the following Python scripts:

1. chatbot4.py: 
   - Implements the main chatbot interface and handles user interactions.
   - Integrates the RAG system with a conversational AI model to generate responses based on user queries.

2. qdrant.py:
   - Contains functions for initializing and managing the Qdrant vector database.
   - Handles collection creation, embedding storage, and vector lookups.

3. Quadrant.py:
   - Responsible for managing the chunking and embedding processes.
   - Uses third-party APIs to generate embeddings for document chunks and stores them in the Qdrant database.

4. rag.py:
   - Implements the core RAG logic, orchestrating the workflow between chunking, embedding, and vector storage.
   - Includes methods for processing input documents, generating embeddings, and handling search queries.

5. rag1.py:
   - An extended version of rag.py, providing additional functionalities or optimizations.
   - Focuses on improving the response generation pipeline with enhanced query processing and context retrieval.

6. test.py:
   - Contains unit tests and integration tests for the components of the RAG system.
   - Ensures that the system behaves as expected and performs efficiently under various scenarios.

7. agentrag.py:
   - Implements the core functionalities of the RAG system, including processing PDF files, chunking text, generating embeddings, and storing data in Qdrant.
   - Handles user queries, retrieves similar text chunks from the vector database, and generates answers using context.
   - Includes question recommendation
   - Includes mechanisms for user feedback and the ability to store user questions for future reference.

Features

- Relevant Chunking: Efficiently splits documents into meaningful chunks for better embedding and retrieval.
- Embedding Generation: Uses third-party LLMs (e.g., Cohere, OpenAI) to generate embeddings for text chunks.
- Vector Storage: Utilizes Qdrant for storing and managing vector embeddings, enabling fast retrieval.
- Vector Lookups: Implements search functionality to find the most relevant chunks based on user queries.
- Response Generation Pipeline: Constructs responses by integrating context from retrieved chunks with user queries using LLMs.
- Question Recommendation: Automatically generates follow-up questions based on the context of the user's queries and the answers provided, enhancing user engagement and exploration of the topic.
- User Satisfaction Feedback Loop: Incorporates user feedback to refine the response generation process, allowing for adjustments in the approach to meet user needs effectively.

All the files can be run using python3 filename.py
