from for_agents_core import generate_answer

SYSTEM_PROMPT = """You are ShopSmart's Escalation Agent.

You handle situations where:
- The customer is highly distressed or angry
- Legal threats or consumer court is mentioned
- Social media escalation is threatened
- The issue has not been resolved after multiple attempts

Your job is to:
1. Deeply acknowledge the customer's frustration with genuine empathy
2. Assure them a senior human specialist will take over immediately
3. Give them a direct escalation contact
4. Never argue, never dismiss, never make promises outside policy

Tone: Warm, calm, senior, deeply empathetic.
Never sound robotic. Never say 'I understand your frustration' — show it instead."""


def escalate(message: str, sentiment: str, client,
             history: list = []) -> dict:

    context = f"""
Customer sentiment level: {sentiment}
This customer requires immediate human escalation.
Acknowledge their experience, apologize genuinely, 
and connect them to a senior specialist.

Escalation contacts:
- Senior Support: 1800-***-**20 (Option 3)
- Grievance Officer: grie*******ce@shopsmart.in
- Response SLA: 48 hours (legally mandated)
"""

    response = generate_answer(
        system_prompt=SYSTEM_PROMPT,
        user_message=message,
        context=context,
        client=client,
        history=history
    )

    return {
        "response":  response,
        "escalated": True,
        "contact":   "grie*******ce@shopsmart.in | 1800-***-**20",
        "agent":     "escalation"
    }