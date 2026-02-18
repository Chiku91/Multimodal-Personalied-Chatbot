import fitz  # PyMuPDF
import docx
import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
from dotenv import load_dotenv
import os
import streamlit as st;

load_dotenv()

# Load Groq API key securely (assumes use of dotenv or st.secrets)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        return " ".join([page.get_text() for page in doc])
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return " ".join([para.text for para in doc.paragraphs])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    else:
        raise ValueError("Unsupported file format")

def split_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def create_vector_store(chunks: List[str]):
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return {"index": index, "chunks": chunks}

def get_relevant_chunks(query: str, vector_store, top_k: int = 5) -> List[str]:
    query_vec = embed_model.encode([query])
    _, indices = vector_store["index"].search(query_vec, top_k)
    return [vector_store["chunks"][i] for i in indices[0]]

def query_groq(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",  # or another supported model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content

def answer_query_with_context(query: str, context_chunks: List[str]) -> str:
    context = "\n".join(context_chunks)
    prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    return query_groq(prompt)


