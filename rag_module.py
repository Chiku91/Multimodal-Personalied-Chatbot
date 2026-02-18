import os
import fitz
import docx
import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=api_key)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        return " ".join([page.get_text() for page in doc])

    if file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return " ".join([p.text for p in doc.paragraphs])

    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    raise ValueError("Unsupported file format")


def split_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)


def create_vector_store(chunks: List[str]):
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return {"index": index, "chunks": chunks}


def get_relevant_chunks(query: str, vector_store, top_k: int = 5):
    query_vec = embed_model.encode([query])
    _, indices = vector_store["index"].search(query_vec, top_k)
    return [vector_store["chunks"][i] for i in indices[0]]


def keyword_search(query: str, chunks: List[str], top_k=5):
    scores = []
    q_words = set(query.lower().split())

    for chunk in chunks:
        overlap = len(q_words & set(chunk.lower().split()))
        scores.append(overlap)

    ranked = sorted(zip(chunks, scores),
                    key=lambda x: x[1],
                    reverse=True)

    return [c for c, _ in ranked[:top_k]]


def generate_query_variants(query: str):
    prompt = f"Generate 3 alternative queries for: {query}"
    res = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    variants = res.choices[0].message.content.split("\n")
    return [query] + variants


def advanced_retrieval(query, vector_store, chunks):
    variants = generate_query_variants(query)

    all_chunks = []
    for q in variants:
        all_chunks.extend(get_relevant_chunks(q, vector_store))

    all_chunks.extend(keyword_search(query, chunks))

    unique = list(dict.fromkeys(all_chunks))
    return unique[:5]


def query_groq(prompt: str):
    res = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    return res.choices[0].message.content


def answer_query_with_context(query: str, context_chunks: List[str]):
    context = "\n".join(context_chunks)

    prompt = f"""
Use the context to answer.

Context:
{context}

Question: {query}
"""

    return query_groq(prompt)


def verify_answer(answer: str, context_chunks):
    context = "\n".join(context_chunks)

    prompt = f"""
Is this answer supported by the context?

Answer:
{answer}

Context:
{context}

Reply YES or NO.
"""

    return query_groq(prompt)
