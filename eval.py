
import os
os.environ["MLFLOW_TRACKING_URI"] = "mlruns"

import time
import mlflow
import mlflow.system_metrics
from datetime import datetime

# MLFlow Setup
EXPERIMENT_NAME = "ShopSmart-Multi-Agent-Support"  

mlflow.set_experiment(EXPERIMENT_NAME)

# Main Tracking Function
def track_conversation(
        query: str,
        intent: str,
        sentiment: str,
        agent: str,
        response: str,
        response_time: float,
        chunks_retrieved: int=0,
        escalated: bool = False,
        ticket: str = "",
):
    with mlflow.start_run(run_name=f"{agent}-{datetime.now()}"):
        # using parms
        mlflow.log_params({
            "intent": intent,
            "sentiment":       sentiment,
            "agent":           agent,
            "escalated":       escalated,
            "has_ticket":      bool(ticket),
            "query_length":    len(query.split()),
        })

        # Needed Metrics

        mlflow.log_metrics({
            "response_time_sec":  round(response_time, 3),
            "response_length":    len(response.split()),
            "chunks_retrieved":   chunks_retrieved,
        })

        # Tags - For filtering in MLflow UI

        mlflow.set_tags({
            "agent_type":   agent,
            "sentiment":    sentiment,
            "intent":       intent,
            "project":      "ShopSmart-Multi-Agent",
            "version":      "1.0.0",
        })

        # Saving the actual conversation

        log_text = f"""
=== ShopSmart Conversation Log ===
Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Intent     : {intent}
Sentiment  : {sentiment}
Agent      : {agent}
Escalated  : {escalated}
Ticket     : {ticket if ticket else 'N/A'}
Chunks     : {chunks_retrieved}
Resp. Time : {round(response_time, 3)}s

--- Query ---
{query}

--- Response ---
{response}
""".strip()
        
        mlflow.log_text(log_text, artifact_file="conversation.txt")

# Batch Evaluation
def run_batch_eval(graph, test_queries: list = None):
    from graph import run_turn

    if test_queries is None:
        test_queries = [
            "Where is my order?",
            "My order has not arrived yet",
            "How long does delivery take?",
            "Do you deliver to Hyderabad?",
            "What are the shipping charges?",
            "Can I change my delivery address?",
            "My order shows delivered but I didn't receive it",
            "Can I schedule a specific delivery time?",
            "Why is my order delayed?",
            "My order was cancelled automatically",
            "Can I order from multiple sellers at once?",
            "What is the estimated delivery date?",
            "Do you deliver on Sundays?",
            "My order is stuck in transit",
            "Can I pick up my order from a store?",

            "I want to return my product",
            "What is your return policy?",
            "How do I initiate a return?",
            "I want a refund for my damaged phone",
            "My refund has not been processed yet",
            "How long does refund take?",
            "I returned my order 10 days ago, where is my refund?",
            "Can I exchange instead of return?",
            "The return pickup was not scheduled",
            "My return request was rejected",
            "Can I return a product after 30 days?",
            "I want to return only one item from my order",
            "How do I return a digital product?",
            "My refund amount is incorrect",
            "Can I get store credit instead of refund?",

            "My payment failed but money was deducted",
            "I was charged twice for the same order",
            "Can I pay using EMI?",
            "Is COD available in my area?",
            "My UPI payment is not going through",
            "Can I split payment between two cards?",
            "My credit card was declined",
            "How do I apply a coupon code?",
            "My cashback was not credited",
            "Can I change payment method after ordering?",
            "Is my payment information secure?",
            "I want to pay using net banking",

            "Is the Samsung Galaxy S24 available?",
            "What is the warranty on electronics?",
            "Does this laptop support Windows 11?",
            "What are the specs of iPhone 15?",
            "Is this product compatible with my device?",
            "Do you sell original Apple products?",
            "What is the difference between these two models?",
            "Is this product available in blue color?",
            "When will this out of stock item be available?",
            "Do you have a product comparison feature?",
            "Can I see a demo before buying?",
            "What brands do you carry for headphones?",

            "You sent me the wrong item",
            "My product arrived damaged",
            "The package was open when it arrived",
            "I received a used product instead of new",
            "Missing accessories in my order",
            "The product quality is very poor",
            "Your delivery person was rude",
            "I was promised a discount that was not applied",
            "My order was delivered to wrong address",
            "The product looks different from the photo",
            "I have been waiting for resolution for 2 weeks",
            "Your customer service is useless",
            "I want to file a formal complaint",

            "This is a scam, I will go to consumer court",
            "I will post about this on Twitter",
            "I am filing a complaint with consumer forum",
            "This is absolutely unacceptable, I want to speak to manager",
            "I have been cheated, I want legal action",
            "Worst experience ever, I will never shop here again",
            "I am extremely angry, you ruined my experience",
            "I will contact TRAI and consumer helpline",
            "I am going to expose this company on social media",
            "My lawyer will be in touch with you",
            "This is fraud and I have evidence",
            "I demand immediate resolution or I go to police",
            "You have lost a customer forever, this is criminal",

            "How do I claim warranty for my laptop?",
            "My phone stopped working within warranty period",
            "Where is the nearest service center?",
            "Is accidental damage covered under warranty?",
            "My warranty card is missing",
            "How long is the warranty on this product?",
            "Can I extend my warranty?",
            "Who do I contact for warranty claims?",

            "I forgot my password",
            "My account has been suspended",
            "I cannot login to my account",
            "How do I delete my account?",
            "I want to update my phone number",
            "My account was hacked",
            "How do I change my email address?",

            "What are your customer support hours?",
            "How do I track my order?",
            "Do you have a loyalty program?",
            "What is ShopSmart?",
            "How do I contact customer support?",
                ]    

    print(f"\n{'='*50}")
    print(f"  ShopSmart Batch Evaluation — {len(test_queries)} queries")
    print(f"{'='*50}\n")

    results = []

    for i, query in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}] {query[:50]}...")

        start = time.time()
        result = run_turn(
            message=query,
            history=[],
            graph=graph
        )  
        end = time.time()    

        response_time = round(end-start,3)

         # The above runs a batch of test queries through the agent system
         # and log all results to MLFlow
        
        # figure out agent
        if result.get("escalated"):
            agent = "escalation"
        elif result.get("ticket"):
            agent = "action"
        else:
            agent = "knowledge"

        # log to MLflow
        track_conversation(
            query          = query,
            intent         = result.get("intent", "unknown"),
            sentiment      = result.get("sentiment", "unknown"),
            agent          = agent,
            response       = result.get("response", ""),
            response_time  = response_time,
            chunks_retrieved = len(result.get("chunks", [])),
            escalated      = result.get("escalated", False),
            ticket         = result.get("ticket", ""),
        )

        results.append({
            "query":         query,
            "intent":        result.get("intent"),
            "sentiment":     result.get("sentiment"),
            "agent":         agent,
            "response_time": response_time,
        })

        print(f" Intent: {result.get('intent')} | "
              f"Agent: {agent} | "
              f"Time: {response_time}s")
    # Summary
    print(f"\n{'='*50}")
    print("  Batch Evaluation Complete")
    print(f"{'='*50}")

    agent_counts = {}
    intent_counts = {}
    total_time = 0

    for r in results:
        agent_counts[r["agent"]]   = agent_counts.get(r["agent"], 0) + 1
        intent_counts[r["intent"]] = intent_counts.get(r["intent"], 0) + 1
        total_time += r["response_time"]

    print(f"\nAgent Distribution:")
    for agent, count in agent_counts.items():
        pct = round(count / len(results) * 100, 1)
        print(f"  {agent:15} → {count} queries ({pct}%)")

    print(f"\nAvg Response Time : {round(total_time / len(results), 3)}s")
    print(f"Total Queries     : {len(results)}")
    print(f"\nView results: mlflow ui")
    print(f"{'='*50}\n")

    return results

if __name__ == "__main__":
    from graph import build_graph
    graph = build_graph()
    run_batch_eval(graph)
