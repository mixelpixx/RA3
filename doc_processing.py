import os
import re
import sqlite3
import logging
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple

# Configuration
DOCUMENTS_PATH = './documents/'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DB_FILE = "knowledge_base.db"

# Preprocessing and Chunking
def preprocess_document(text):
  # Markdown-specific cleaning
  clean_text = re.sub(r"([_*#`\[\]])|(\n|\t)", " ", text)  

  # Basic cleaning (you might retain some of this)
  clean_text = re.sub(r"[^\w\s\.]", "", clean_text)  

  sentences = sent_tokenize(clean_text)
  return sentences

def chunk_documents(documents_path: str) -> List[Tuple[str, str]]:
  chunks = []
  for file_name in os.listdir(documents_path):
    if file_name.endswith(".txt"):
      try:
        with open(os.path.join(documents_path, file_name), 'r', encoding='utf-8') as file:
          text = file.read()
          preprocessed_sentences = preprocess_document(text)
      except Exception as e:
        logging.error(f"Error processing file {file_name}: {e}")
        continue
      chunks.extend([(file_name, sentence) for sentence in preprocessed_sentences])
  return chunks

def generate_embeddings(chunks: List[Tuple[str, str]]) -> np.ndarray:
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    return embeddings

def create_faiss_index(embeddings, index_file="faiss_index.index"):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_file)
    return index

# Database Functions
def create_database():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                file_name TEXT,
                text TEXT,
                embeddings BLOB
            )
        ''')
        conn.commit()
    except Exception as e:
        logging.error(f"Error creating database: {e}")
    finally:
        conn.close()
  
def save_embeddings(chunks: List[Tuple[str, str]], embeddings: np.ndarray):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.executemany(
            "INSERT INTO documents (file_name, text, embeddings) VALUES (?, ?, ?)",
            [(chunk[0], chunk[1], embedding.tobytes()) for chunk, embedding in zip(chunks, embeddings)]
        )
        conn.commit()
    except Exception as e:
        logging.error(f"Error saving embeddings: {e}")
    finally:
        conn.close()

def load_embeddings(index_file="faiss_index.index"):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT file_name, embeddings FROM documents")
        results = cursor.fetchall()
    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        return None, None, None
    finally:
        conn.close()

    embeddings = np.stack([np.frombuffer(result[1], dtype=np.float32) for result in results])
    index = faiss.read_index(index_file)
    file_names = [result[0] for result in results]  # Load associated file names

    return index, embeddings, file_names
