# Medical_chatbot

Medical Chatbot using RAG & LangChain
This project is a Medical Question-Answering Chatbot built using Streamlit, LangChain, and OpenAI. It implements RAG (Retrieval-Augmented Generation), enabling accurate and contextual responses from your own medical documents (PDFs).

Features
Retrieval-Augmented Generation: Combine LLM power with domain-specific PDFs.

PDF-based QA: Ask any question based on your uploaded medical literature.

Persistent Vector Database: Saves embeddings using Chroma for faster reuse.

HuggingFace Embeddings: Uses all-MiniLM-L6-v2 for semantic search.

Streamlit UI: Clean medical-themed interface for smooth interaction.

.env Support: Securely manage your OpenAI API key.

How It Works
Upload medical PDFs into the ./Data folder.

Documents are loaded and split into small chunks.

Embeddings are generated using HuggingFace transformers.

Stored in a persistent Chroma vector database.

User asks a medical question.

The most relevant chunks are retrieved and sent to OpenAI for an answer.



