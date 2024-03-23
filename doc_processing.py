import os
import re
import sqlite3
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss

# Configuration
DOCUMENTS_PATH = './documents/'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(EMBEDDING_MODEL)
DB_FILE = "knowledge_base.db"

# Preprocessing and Chunking
def preprocess_document(text):
  # Markdown-specific cleaning
  clean_text = re.sub(r"([_*#`\[\]])|(\n|\t)", " ", text)  

  # Basic cleaning (you might retain some of this)
  clean_text = re.sub(r"[^\w\s\.]", "", clean_text)  

  sentences = sent_tokenize(clean_text)
  return sentences

def chunk_documents(documents_path):
  chunks = []
  for file_name in os.listdir(documents_path):
    if file_name.endswith(".txt"):
      with open(os.path.join(documents_path, file_name), 'r', encoding='utf-8') as file:
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
            text TEXT,  -- Full text is now stored
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
