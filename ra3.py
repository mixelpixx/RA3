import argparse  # For command-line arguments
import faiss
import os
import openai
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from openai import OpenAI

client = OpenAI()

# Retrieve OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Configuration
DOCUMENTS_PATH = './documents/'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(EMBEDDING_MODEL)
DB_FILE = "knowledge_base.db"

# Preprocessing and Chunking
def preprocess_document(text):
  sentences = sent_tokenize(text)
  return sentences

def chunk_documents(documents_path):
  chunks = []
  for file_name in os.listdir(documents_path):
    if file_name.endswith(".txt"):
      with open(os.path.join(documents_path, file_name), 'r') as file:
        text = file.read()
        preprocessed_sentences = preprocess_document(text)
        chunks.extend(preprocessed_sentences)
  return chunks

# Embedding Generation
def generate_embeddings(chunks):
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    return embeddings

# FAISS Index Creation
def create_faiss_index(embeddings, index_file="faiss_index.index"):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_file)
    return index

# Database Functions
def create_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            file_name TEXT,
            text TEXT,  # Full text is now stored
            embeddings BLOB
        )
    ''')
    conn.commit()
    conn.close()
  
def save_embeddings(file_names, embeddings, chunks):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for file_name, embedding, chunk in zip(file_names, embeddings, chunks):
        cursor.execute(
            "INSERT INTO documents (file_name, embeddings, text) VALUES (?, ?, ?)",
            (file_name, embedding.tobytes(), chunk)
        )
    conn.commit()
    conn.close()

def load_embeddings(index_file="faiss_index.index"):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, embeddings FROM documents")
    results = cursor.fetchall()
    conn.close()

    embeddings = np.stack([np.frombuffer(result[1], dtype=np.float32) for result in results])
    index = faiss.read_index(index_file)
    file_names = [result[0] for result in results]  # Load associated file names

    return index, embeddings, file_names

def knowledge_query(query: str, index: faiss.Index, chunks: list, file_names, top_k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k=top_k)

    # Filter and retrieve relevant information based on indices
    top_results = [(chunks[idx], file_names[idx]) for idx in indices[0]] 
    return top_results
    
def keyword_search(query, top_n=10):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT file_name, text FROM documents WHERE file_name LIKE ? OR text LIKE ?", 
        (f"%{query}%", f"%{query}%")
    )
    results = cursor.fetchall()[:top_n]
    conn.close()
    return results 

def answer_question(question, index, chunks, file_names):
    # Step 1: Keyword Search for Initial Filtering
    initial_results = keyword_search(question)

    # Step 2: Semantic Ranking with FAISS (if we have initial results)
    if initial_results:
        relevant_chunks = [result[1] for result in initial_results]
        relevant_embeddings = generate_embeddings(relevant_chunks)  
        temp_index = create_faiss_index(relevant_embeddings)  
        # Limit the results based on the length of relevant chunks
        top_results = knowledge_query(question, temp_index, relevant_chunks, initial_results[:len(relevant_chunks)][0])

        # Step 3: Answer Generation with OpenAI
        retrieved_text = "\n\n".join(result[0] for result in top_results)  # Format retrieved chunks
        try:
            openai.api_key = OPENAI_API_KEY  # Ensure the API key is set
            response = openai.ChatCompletion.create(  # Use ChatCompletion for better context handling
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": retrieved_text},
                    {"role": "user", "content": question}
                ]
            )
            return response.choices[0].message["content"]

        except Exception as e:
            return f"An error occurred while generating the answer: {e}" 

    else:
        return "No relevant documents found. Try refining your query."  # Graceful fallback
 

if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Knowledge base question answering system")
    parser.add_argument('--init', action='store_true', 
                        help='Initialize knowledge base (preprocess docs, embeddings, etc.)')
    args = parser.parse_args()

    try:
        if args.init:  # Initialization Mode

            # Step 1: Preprocess Documents
            chunks = chunk_documents(DOCUMENTS_PATH)

            # Step 2: Generate Embeddings
            embeddings = generate_embeddings(chunks)

            # Step 3: Create FAISS index
            index = create_faiss_index(embeddings)

            # Step 4: Store Data in the Database
            create_database()  
            file_names = os.listdir(DOCUMENTS_PATH)  # Get a fresh list 
            save_embeddings(file_names, embeddings, chunks)  

            print("Knowledge base initialization complete!")

        else:  # Question Answering Mode
            if not os.path.exists("faiss_index.index") or not os.path.exists(DB_FILE):
                raise FileNotFoundError("Index and/or database not found. Run with '--init' first.")

            index, embeddings, file_names = load_embeddings()

            while True:
                question = input("Enter your question (or type 'exit' to quit): ").strip()
                if question.lower() == 'exit':
                    break

                try:
                    answer = answer_question(question, index, file_names, embeddings)
                    print(f"Answer: {answer}")
                except Exception as e:
                    print(f"An error occurred: {e}")

    except Exception as general_error:
        print(f"A critical error occurred: {general_error}")