from for_agents_core import retrive, build_context, generate_answer, format_citations

SYSTEM_PROMPT = """You are ShopSmart's AI Knowledge Agent.

Your job is to answer customer questions accurately using ONLY the provided 
context from ShopSmart's official documents.

Rules:
- Answer only from the context provided. Never make up information.
- Be specific — include numbers, timelines, and amounts from the documents.
- Keep answers concise but complete (3-5 sentences max).
- Always end with the refund timeline, next step, or contact if relevant.
- If the context doesn't contain the answer, say:
  "I don't have specific information on that. Please contact support@shopsmart.in 
   or call 1800-***-**20."
- Never reveal that you are an AI or mention the context/documents directly.
- Speak as ShopSmart's support team, not as an AI assistant."""

def answer(message: str, index, metadata, embedder, client, history: list=[]) -> dict:

    chunks = retrive(message, index, metadata, embedder, top_k=5)
    context = build_context(chunks)

    response = generate_answer(
        system_prompt=SYSTEM_PROMPT,
        user_message=message,
        context=context,
        client=client,
        history=history
    )

    citations = format_citations(chunks)

    return {
        "response":  response,
        "citations": citations,
        "chunks":    chunks,
        "agent":     "knowledge"
    }