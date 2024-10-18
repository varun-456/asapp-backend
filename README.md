Agentrag
Agentrag is a Python-based application designed to process PDF files, extract text, chunk it, and generate embeddings using Cohere's embedding model. These embeddings are stored in Qdrant Cloud for vector-based semantic search. The system also supports querying for similar document chunks and generating contextual answers based on user queries. It includes functionality to store and recommend follow-up questions to improve user interaction.
Features
•	PDF Text Extraction: Extracts text from PDF files using pdfplumber.
•	Text Chunking: Chunks text into manageable segments using Langchain's RecursiveCharacterTextSplitter.
•	Embedding Generation: Uses Cohere's API to generate 4096-dimensional embeddings.
•	Semantic Search: Allows querying for semantically similar document chunks using Qdrant Cloud.
•	Answer Generation: Generates contextual answers to user questions using retrieved document chunks.
•	Follow-Up Question Recommendation: Recommends potential follow-up questions based on generated answers.
Prerequisites
•	Python 3.7+
•	API keys for:
o	Cohere
o	Qdrant Cloud
