from for_agents_core import load_components, detect_sentiment
from agents.classifier import classify
from agents.knowledge import answer
from agents.actions import process
from agents.escalations import escalate

from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class SupportState(TypedDict):
    message:   str          
    intent:    str           
    sentiment: str           
    response:  str           
    citations: str           
    ticket:    str          
    escalated: bool          
    history:   List[dict]
    agent: str


index, metadata, embedder, client = load_components()

# Graph Node - Functions

def node_classify(state: SupportState) -> SupportState:
    message = state["message"]
    intent = classify(message, index, metadata, embedder, client)
    sentiment = detect_sentiment(message, client)
    print(f"  => Intent : {intent} || Sentiment : {sentiment}")
    return {**state, "intent": intent, "sentiment": sentiment}

def node_knowledge(state: SupportState) -> SupportState:
    result = answer(
        state["message"], index, metadata,
        embedder, client, state["history"]
    )
    return {
        **state,
        "response":  result["response"],
        "citations": result["citations"],
        "agent":     "knowledge", 
    }

def node_action(state: SupportState) -> SupportState:
    result = process(
        state["message"], state["intent"],
        client, state["history"]
    )
    return {
        **state,
        "response": result["response"],
        "ticket":   result["ticket"],
        "agent":    "action", 
    }

def node_escalation(state: SupportState) -> SupportState:
    result = escalate(
        state["message"], state["sentiment"],
        client, state["history"]
    )
    return {
        **state,
        "response":  result["response"],
        "escalated": result["escalated"],
        "agent":     "escalation",
    }


# Main == Router to decide agent 

def router(state: SupportState) -> str:
    sentiment = state["sentiment"]
    intent = state["intent"]

    if sentiment == "HIGH_DISTRESS":
        return "escalation"
    
    if intent in [
        "return_request", "refund_status",
                  "payment_issue", "order_status","complaint",
                ]:
        return "action"
    
    if intent == "escalation":
        return "escalation"
    
    return "knowledge"

# Main Graph - Bulding the Graph

def build_graph():
    graph = StateGraph(SupportState)
    # adding nodes
    graph.add_node("classifier",node_classify)
    graph.add_node("knowledge",  node_knowledge)
    graph.add_node("action",     node_action)
    graph.add_node("escalation", node_escalation)

    # What's the entry point ?? yeah - classifier
    graph.set_entry_point("classifier")

    # After classifier; router decides the nxt node
    graph.add_conditional_edges(
        "classifier",
        router,
        {
            "knowledge":  "knowledge",
            "action":     "action",
            "escalation": "escalation",
        }
    )

    # All agents; ends after responding
    graph.add_edge("knowledge",END)
    graph.add_edge("action",     END)
    graph.add_edge("escalation", END)

    return graph.compile()

def run_turn(message: str, history : List[dict], graph) -> dict:
    initial_state = SupportState(
        message   = message,
        intent    = "",
        sentiment = "",
        response  = "",
        citations = "",
        ticket    = "",
        escalated = False,
        history   = history,
        agent="",
    )

    result = graph.invoke(initial_state)
    return result


