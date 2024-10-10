import argparse  # For command-line arguments
import faiss
import os
import sys
import logging
import getpass
import openai
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from openai import OpenAI
from doc_processing import preprocess_document, chunk_documents, generate_embeddings, create_faiss_index, create_database, save_embeddings, load_embeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="RAG Chatbot")
parser.add_argument("--rebuild", action="store_true", help="Rebuild the knowledge base")
args = parser.parse_args()

# Configuration
DOCUMENTS_PATH = './documents/'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DB_FILE = "knowledge_base.db"
FAISS_INDEX_FILE = "faiss_index.index"
MAX_CONTEXT_LENGTH = 4096

client = OpenAI()

# Initialize sentence transformer
embedder = SentenceTransformer(EMBEDDING_MODEL)

def semantic_search(query: str, index: faiss.Index, embeddings: np.ndarray, file_names: List[str], k: int = 5) -> List[Tuple[str, str]]:
    """
    Perform semantic search on the document embeddings.
    """
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    results = []
    for idx in indices[0]:
        file_name = file_names[idx]
        text = get_document_text(file_name)
        results.append((file_name, text))
    return results

def get_document_text(file_name: str) -> str:
    """
    Retrieve the full text of a document from the database.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT text FROM documents WHERE file_name = ?", (file_name,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else ""

def generate_response(query: str, context: List[Tuple[str, str]], conversation_history: List[dict]) -> str:
    """
    Generate a response using GPT-3.5-turbo based on the query, context, and conversation history.
    """
    system_message = "You are a helpful AI assistant with access to a knowledge base. Use the provided context to answer the user's questions."
    messages = [{"role": "system", "content": system_message}]
    
    # Add conversation history
    messages.extend(conversation_history[-5:])  # Include last 5 exchanges
    
    # Add context from semantic search
    context_text = "\n\n".join([f"File: {file_name}\nContent: {text}" for file_name, text in context])
    messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nUser Query: {query}"})
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "I'm sorry, but I encountered an error while generating a response. Please try again."

def main():
    if args.rebuild:
        logging.info("Rebuilding knowledge base...")
        create_database()
        chunks = chunk_documents(DOCUMENTS_PATH)
        embeddings = generate_embeddings(chunks)
        file_names = [os.path.basename(f) for f in os.listdir(DOCUMENTS_PATH) if f.endswith('.txt')]
        save_embeddings(chunks, embeddings)
        create_faiss_index(embeddings, FAISS_INDEX_FILE)
        logging.info("Knowledge base rebuilt successfully.")

    logging.info("Loading embeddings and FAISS index...")
    index, embeddings, file_names = load_embeddings(FAISS_INDEX_FILE)
    logging.info("Embeddings and FAISS index loaded successfully.")

    conversation_history = []
    
    print("Welcome to the RAG Chatbot! Type 'quit' to exit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            break

        search_results = semantic_search(user_input, index, embeddings, file_names)
        response = generate_response(user_input, search_results, conversation_history)

        print(f"AI: {response}")

        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})

        # Trim conversation history if it gets too long
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]

if __name__ == "__main__":
    main()
