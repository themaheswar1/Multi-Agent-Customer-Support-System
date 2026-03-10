
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
            "Where is my order ?, Track it & update me",
            "What is your return policy & precautions",
            "I want to return my damaged phone and get refund for that !!",
            "My payment failed but money was deducted",
            "This is a scam, I will go to consumer court",
            "Do you deliver to Hyderabad?",
            "I want a refund for order #12345",
            "What is the warranty on electronics?",
            "Wrong item was delivered, I want exchange",
            "I am extremely angry, you ruined my experience",
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
