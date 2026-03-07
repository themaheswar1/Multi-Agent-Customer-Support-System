from for_agents_core import retrive, build_context, generate_answer

INTENTS = [
    "order_status",      
    "return_request",    
    "refund_status",     
    "product_query",     
    "payment_issue",     
    "complaint",         
    "shipping_query",    
    "warranty_claim",    
    "account_issue",     
    "escalation",        
    "general",           
]

SYSTEM_PROMPT = """You are a customer intent classifier for ShopSmart E-Commerce.

Your ONLY job is to classify the customer's message into exactly one intent.

Available intents:
- order_status      → tracking, order not received, delivery date
- return_request    → want to return, return policy, exchange
- refund_status     → refund not received, refund delay, refund amount
- product_query     → product details, specs, stock, compatibility
- payment_issue     → payment failed, charged twice, EMI, COD
- complaint         → damaged item, wrong item, missing item, bad experience
- shipping_query    → delivery time, shipping cost, pin code check
- warranty_claim    → warranty, repair, service center, replacement
- account_issue     → login problem, account suspended, password reset
- escalation        → legal threats, consumer court, social media threat
- general           → greetings, feedback, anything else

Reply with ONLY the intent name. No explanation. No punctuation. Just the intent."""


def classify(message: str, index, metadata, embedder, client) -> str:
    chunks = retrive(message, index, metadata, embedder, top_k=3)
    context = build_context(chunks)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nCustomer message: {message}"}
        ],
        temperature=0.0,
        max_tokens=10,
    )

    intent = response.choices[0].message.content.strip().lower()

    return intent if intent in INTENTS else "general" # Safety fallback