import os
import fitz
import docx
import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=api_key)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------- TEXT EXTRACTION ----------
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


# ---------- CHUNKING ----------
def split_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)


# ---------- VECTOR STORE ----------
def create_vector_store(chunks: List[str]):
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return {"index": index, "chunks": chunks}


def get_relevant_chunks(query: str, vector_store, top_k: int = 5):
    query_vec = embed_model.encode([query])
    _, indices = vector_store["index"].search(query_vec, top_k)
    return [vector_store["chunks"][i] for i in indices[0]]


# ---------- KEYWORD SEARCH ----------
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


# ---------- MULTI-QUERY ----------
def generate_query_variants(query: str):
    prompt = f"Generate 3 alternative queries for: {query}"
    res = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    variants = res.choices[0].message.content.split("\n")
    return [query] + variants


# ---------- HYBRID RETRIEVAL ----------
def advanced_retrieval(query, vector_store, chunks):
    variants = generate_query_variants(query)

    all_chunks = []
    for q in variants:
        all_chunks.extend(get_relevant_chunks(q, vector_store))

    all_chunks.extend(keyword_search(query, chunks))

    unique = list(dict.fromkeys(all_chunks))
    return unique[:5]


# ---------- GROQ QUERY ----------
def query_groq(prompt: str):
    res = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    return res.choices[0].message.content


# ---------- ANSWER ----------
def answer_query_with_context(query: str, context_chunks: List[str]):
    context = "\n".join(context_chunks)

    prompt = f"""
Use the context to answer.

Context:
{context}

Question: {query}
"""
    return query_groq(prompt)


# ---------- SELF-VERIFICATION ----------
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


def evaluate_answer(answer: str, context_chunks):

    context = " ".join(context_chunks)

    emb1 = embed_model.encode(answer, convert_to_tensor=True)
    emb2 = embed_model.encode(context, convert_to_tensor=True)

    similarity = util.cos_sim(emb1, emb2).item()

    ans_words = set(answer.lower().split())
    ctx_words = set(context.lower().split())

    overlap = len(ans_words & ctx_words)

    faithfulness = overlap / max(len(ans_words), 1)
    coverage = overlap / max(len(ctx_words), 1)
    confidence = (similarity + faithfulness) / 2
    hallucination_risk = 1 - faithfulness

    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score_val = rouge.score(context, answer)['rougeL'].fmeasure

    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu(
        [context.split()],
        answer.split(),
        smoothing_function=smoothie
    )

    quality = np.mean([
        similarity,
        faithfulness,
        coverage,
        rouge_score_val,
        bleu
    ])

    return {
        "similarity": similarity,
        "faithfulness": faithfulness,
        "coverage": coverage,
        "confidence": confidence,
        "hallucination_risk": hallucination_risk,
        "rougeL": rouge_score_val,
        "bleu": bleu,
        "quality_score": quality
    }


# ======================================================
# 📈 MULTIMODAL ACCURACY METRICS
# ======================================================

def evaluate_multimodal_response(
    query: str,
    answer: str,
    context_chunks=None,
    input_mode="Text",
    extracted_text=None,
    intent=None
):

    metrics = {}

    emb_q = embed_model.encode(query, convert_to_tensor=True)
    emb_a = embed_model.encode(answer, convert_to_tensor=True)

    relevance = util.cos_sim(emb_q, emb_a).item()
    metrics["relevance"] = relevance

    metrics["consistency"] = relevance

    if context_chunks:
        context = " ".join(context_chunks)
        emb_ctx = embed_model.encode(context, convert_to_tensor=True)
        grounded = util.cos_sim(emb_a, emb_ctx).item()
        metrics["grounded_accuracy"] = grounded
    else:
        metrics["grounded_accuracy"] = None

    if extracted_text:
        emb_ext = embed_model.encode(extracted_text, convert_to_tensor=True)
        modality_score = util.cos_sim(emb_ext, emb_a).item()
        metrics["modality_reliability"] = modality_score
    else:
        metrics["modality_reliability"] = None

    if intent:
        metrics["intent_correctness"] = 1.0
    else:
        metrics["intent_correctness"] = None

    valid_scores = [
        v for v in metrics.values()
        if isinstance(v, (float, int)) and v is not None
    ]

    overall_accuracy = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    metrics["overall_accuracy"] = overall_accuracy

    return metrics
