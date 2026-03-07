import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Config for path usage
FAISS_INDEX_PATH = "vectorstore/index.faiss"
METADATA_PATH = "vectorstore/metadata.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL       = "llama-3.3-70b-versatile"
TOP_K            = 5 

# componenets loading 

def load_components():
    
    index = faiss.read_index(FAISS_INDEX_PATH)
    print(" === Loaded FAISS Index === ")

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    client = Groq(api_key=os.getenv("GROQ_API_KEY")) 

    print("=== Core components for agents are loaded ===")

    return index, metadata, embedder, client

# Retriving Relevant Chunks

def retrive(query: str, index, metadata, embedder, top_k: int = TOP_K):
    query_vec = embedder.encode([query], convert_to_numpy = True) # True coz, faiss works with np 
    norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
    query_vec = query_vec / np.maximum(norm, 1e-10)

    scores, indices = index.search(query_vec.astype(np.float32), top_k)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk_meta = metadata[idx]
        results.append({
            "text": chunk_meta["text"],
            "source":    chunk_meta["source"],
            "file_type": chunk_meta["file_type"],
            "page":      chunk_meta["page"],
            "score":     float(score)
        })

    return results

# Building Context String from chunks we retrive

def build_context(chunks: list) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Source {i}: {chunk['source']} | "
            f"Type: {chunk['file_type'].upper()} | "
            f"Page/Row: {chunk['page']} | "
            f"Relevance: {chunk['score']:.2f}]\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(context_parts)

# Generating Answer using Groq

def generate_answer(
        system_prompt : str, user_message: str,
        context: str, client,
        history: list = []
        ) -> str:
    # building msgs
    messages = [{"role": "system", "content": system_prompt}]
    #converstation history down
    for turn in history[-6:]:
        messages.append(turn)

    # Adding current query with context
    messages.append({
        "role":"user",
        "content": f"Context from ShopSmart knowledge base:\n{context}\n\nCustomer query: {user_message}"
    })

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,      
        max_tokens=512,
    )

    return response.choices[0].message.content.strip()

# Detecting Customer Sentiment

def detect_sentiment(message: str, client) -> str:
    prompt = """You are a Sentiment classifier for customer support messages.
    Classigy the sentiment as exactly to the one of: POSITIVE, NEUTRAL, NEGATIVE, HIGH_DISTRESS.

    HIGH_DISTRESS = threats of legal action, extreme anger, mentions of fraud/scam,
                social media threats, repeated complaints.
    NEGATIVE      = frustrated, unhappy, disappointed.
    NEUTRAL       = asking questions, seeking information.
    POSITIVE      = happy, satisfied, complimenting.

Reply with ONLY the single word. Nothing else. 

"""
    response = client.chat.completions.create(
        model= GROQ_MODEL,
        messages = [
            {"role":"system", "content": prompt},
            {"role":"user","content":message}
        ],
        temperature=0.0,
        max_tokens=10,
    )
    sentiment = response.choices[0].message.content.strip().upper()

    #safety fallback
    valid = {"POSITIVE", "NEUTRAL", "NEGATIVE", "HIGH_DISTRESS"}
    return sentiment if sentiment in valid else "NEUTRAL"

# Format Citation

def format_citations(chunks: list) -> str:
    seen = set()
    citations = []
    for chunk in chunks:
        key = chunk["source"]
        if key not in seen:
            seen.add(key)
            citations.append(
                f" {chunk['source']}"
                f"(Page/Row {chunk['page']})"
            )
    return "\n".join(citations)        

    







