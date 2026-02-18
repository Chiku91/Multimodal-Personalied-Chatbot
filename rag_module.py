import fitz  # PyMuPDF
import docx
import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
import os
import streamlit as st


api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
groq_client = Groq(api_key=api_key)


embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        return " ".join([page.get_text() for page in doc])

    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return " ".join([para.text for para in doc.paragraphs])

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError("Unsupported file format")


def split_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
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


def keyword_search(query: str, chunks: List[str], top_k=5):
    scores = []
    query_words = set(query.lower().split())

    for chunk in chunks:
        overlap = len(query_words & set(chunk.lower().split()))
        scores.append(overlap)

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]

def hybrid_retrieval(query, vector_store, chunks, top_k=5):
    semantic = get_relevant_chunks(query, vector_store, top_k)
    keyword = keyword_search(query, chunks, top_k)

    combined = list(dict.fromkeys(semantic + keyword))
    return combined[:top_k]

def generate_query_variants(query: str) -> List[str]:
    prompt = f"""
Generate 3 alternative search queries that mean the same as:
"{query}"
Only output the queries, one per line.
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.choices[0].message.content
    variants = [q.strip() for q in text.split("\n") if q.strip()]

    return [query] + variants

def advanced_retrieval(query, vector_store, chunks):
    variants = generate_query_variants(query)

    all_chunks = []

    for q in variants:
        all_chunks.extend(get_relevant_chunks(q, vector_store))

    all_chunks.extend(keyword_search(query, chunks))

    unique = list(dict.fromkeys(all_chunks))
    return unique[:5]


def query_groq(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content


def answer_query_with_context(query: str, context_chunks: List[str]) -> str:
    context = "\n".join(context_chunks)

    prompt = f"""
You are an AI tutor.

Use ONLY the context below to answer.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

    return query_groq(prompt)
