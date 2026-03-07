from for_agents_core import generate_answer
import hashlib

SYSTEM_PROMPT = """You are ShopSmart's Action Agent.

You handle customer requests that require processing actions like:
- Refund status checks
- Order cancellations  
- Return initiations
- Payment issue resolutions

Since you are a demo system, simulate the action professionally.
Respond as if you have checked the system and are reporting results.

Rules:
- Always acknowledge the specific action requested
- Give a clear status and next step with a realistic timeline
- Include relevant contact details when needed
- Be empathetic but efficient
- Sign off with a ticket/reference number (simulate it)

Format: 
1. Action taken / status found
2. What happens next + timeline
3. Reference number"""

def process(message: str, intent: str, client, history : list = []) -> dict:

    action_context = {
        "return_request": "Customer wants to initiate a return. Guide them through the return process.",
        "refund_status":  "Customer is asking about refund status. Check and report status.",
        "payment_issue":  "Customer has a payment problem. Investigate and resolve.",
        "order_status":   "Customer wants order status. Retrieve and report current status.",
    }.get(intent,"Customer needs action on their account. Assist appropriately.")

    response = generate_answer(
        system_prompt=SYSTEM_PROMPT,
        user_message=message,
        context=action_context,
        client=client,
        history=history
    )

    ticket = "SS-"+ hashlib.md5(message.encode()).hexdigest()[:6].upper()

    return{
        "response":  response + f"\n\nYour ticket reference for our conversation: **{ticket}**",
        "ticket":    ticket,
        "agent":     "action"
    }
