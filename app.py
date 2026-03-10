import streamlit as st
from graph import build_graph, run_turn

st.set_page_config(
    page_title="Mahesh\'s ShopSmart Support",
    page_icon="🛍️",
    layout="centered"
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .agent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: bold;
        margin-bottom: 6px;
    }
    .badge-knowledge  { background-color: #1a3a5c; color: #7ec8e3; }
    .badge-action     { background-color: #1a4a2e; color: #7ed99a; }
    .badge-escalation { background-color: #4a1a1a; color: #e37e7e; }
    .citation-box {
        background-color: #1e2130;
        border-left: 3px solid #2e6da4;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 12px;
        color: #8899aa;
        margin-top: 8px;
    }
    .sentiment-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 10px;
        margin-left: 8px;
    }
    .pill-positive      { background-color: #1a4a2e; color: #7ed99a; }
    .pill-neutral       { background-color: #2a2a2a; color: #aaaaaa; }
    .pill-negative      { background-color: #4a3a1a; color: #e3c07e; }
    .pill-high_distress { background-color: #4a1a1a; color: #e37e7e; }

    /* agent card in sidebar */
    .agent-card {
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid #1e1e1e;
        font-size: 13px;
    }
    .agent-card.active {
        border: 1px solid #2e6da4;
        background-color: #0d1f35;
    }
    .agent-card .dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .dot-idle     { background-color: #333; }
    .dot-active   { background-color: #4aca7e;
                    animation: blink 1s infinite; }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.2; }
    }
    .agent-name   { font-weight: bold; color: #ccc; }
    .agent-desc   { font-size: 11px; color: #555; margin-top: 2px; }
    .active .agent-name { color: #7ec8e3; }
    .active .agent-desc { color: #4a7a9a; }
            /* user bubble */
    [data-testid="stChatMessageContent"] {
        background: #1a1a2e !important;
        border: 1px solid #2a2a4a !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
    }

    /* assistant bubble */
    [data-testid="stChatMessage"] {
        padding: 8px 0 !important;
    }
    section[data-testid="stSidebar"] {
    background: #0d0d1a !important;
    border-right: 1px solid #1a1a2e !important;
    padding-top: 1rem !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        font-size: 13px !important;
        color: #888 !important;
        line-height: 1.7 !important;
    }

    section[data-testid="stSidebar"] h3 {
        font-size: 11px !important;
        color: #444 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] {
    font-size: 12px !important;
    }

    section[data-testid="stSidebar"] li {
        font-size: 12px !important;
        line-height: 1.6 !important;
        color: #777 !important;
    }

    section[data-testid="stSidebar"] .stButton button {
        font-size: 11px !important;
        padding: 4px 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("## 🛍️ Mahesh\'s ShopSmart Customer Support")
st.caption("*Multi-Agent AI - Ask anything about your order, returns, payments & more*")
st.caption("4-agent AI system · Knowledge · Action · Escalation · Sentiment detection")
st.divider()

#Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

if "graph" not in st.session_state:
    with st.spinner(""):
        status = st.status("🚀 Starting ShopSmart AI System...", expanded=True)
        with status:
            st.write(" == Loading FAISS vector index...")
            st.write(" == Loading embedding model...")
            st.write(" == Connecting to Groq LLM...")
            st.write(" == Building LangGraph pipeline...")
            st.write(" == Wiring 4 agents together...")
            st.session_state.graph = build_graph()
            status.update(
                label=" == All systems ready — 4 agents online",
                state="complete",
                expanded=False
            )

if "active_agent" not in st.session_state:
    st.session_state.active_agent = None

with st.sidebar:
    st.markdown("### Agent System")
    st.caption("Last active agent is highlighted")
    st.markdown("")

    agents = [
        {
            "key":   "knowledge",
            "icon":  "🧠",
            "name":  "Knowledge Agent",
            "desc":  "Policies · FAQs · Product info"
        },
        {
            "key":   "action",
            "icon":  "⚙️",
            "name":  "Action Agent",
            "desc":  "Orders · Returns · Refunds"
        },
        {
            "key":   "escalation",
            "icon":  "🚨",
            "name":  "Escalation Agent",
            "desc":  "Complaints · Legal · High distress"
        },
    ] 

    for ag in agents:
        is_active = st.session_state.active_agent == ag["key"]
        card_class = "agent-card active" if is_active else "agent-card"
        dot_class = "dot dot-active" if is_active else "dot dot-idle"
        status_text = " Running last turn" if is_active else ""

        st.markdown(f"""
        <div class="{card_class}">
            <span class="{dot_class}"></span>
            <span class="agent-name">{ag["icon"]} {ag["name"]}</span>
            <div class="agent-desc">{ag["desc"]}</div>
            {"<div style='font-size:10px;color:#2e6da4;margin-top:4px;'>▶ handled last message</div>" if is_active else ""}
        </div>
        """, unsafe_allow_html=True)  

    st.divider()
    st.markdown("### Examples ? Try These")
    st.markdown("""
- What is your return policy?
- I want a refund for my damaged phone
- This is a scam, I'll go to consumer court.
- Do you deliver to Hyderabad?
""")
    st.divider()

    if st.button("Clear Conversation"):
        st.session_state.messages      = []
        st.session_state.history       = []
        st.session_state.active_agent  = None
        st.rerun()

    st.markdown("### 📊 Session Stats")
    st.markdown(f"Messages: **{len(st.session_state.messages)}**")

# ── Render Past Messages ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            agent     = msg.get("agent", "knowledge")
            sentiment = msg.get("sentiment", "neutral").lower()

            st.markdown(
                f'<span class="agent-badge badge-{agent}">'
                f'{"🧠 Knowledge" if agent == "knowledge" else "⚙️ Action" if agent == "action" else "🚨 Escalation"}'
                f' Agent</span>'
                f'<span class="sentiment-pill pill-{sentiment}">'
                f'{sentiment.upper()}</span>',
                unsafe_allow_html=True
            )
            st.markdown(msg["content"])

            if msg.get("citations"):
                st.markdown(
                    f'<div class="citation-box">📚 Sources:<br>{msg["citations"]}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(msg["content"])

# ── Chat Input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Type your question here..."):

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role":    "user",
        "content": prompt
    })

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            result = run_turn(
                message = prompt,
                history = st.session_state.history,
                graph   = st.session_state.graph
            )

        # figure out which agent responded
        if result.get("escalated"):
            agent = "escalation"
        elif result.get("ticket"):
            agent = "action"
        else:
            agent = "knowledge"

        sentiment = result.get("sentiment", "neutral").lower()
        response  = result.get("response", "I'm sorry, I couldn't process that.")
        citations = result.get("citations", "")

        st.markdown(
            f'<span class="agent-badge badge-{agent}">'
            f'{"🧠 Knowledge" if agent == "knowledge" else "⚙️ Action" if agent == "action" else "🚨 Escalation"}'
            f' Agent</span>'
            f'<span class="sentiment-pill pill-{sentiment}">'
            f'{sentiment.upper()}</span>',
            unsafe_allow_html=True
        )
        st.markdown(response)

        if citations:
            st.markdown(
                f'<div class="citation-box">📚 Sources:<br>{citations}</div>',
                unsafe_allow_html=True
            )

    # save to session
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   response,
        "agent":     agent,
        "sentiment": sentiment,
        "citations": citations,
    })

    # track to MLflow
    from eval import track_conversation
    track_conversation(
        query          = prompt,
        intent         = result.get("intent", "unknown"),
        sentiment      = sentiment,
        agent          = agent,
        response       = response,
        response_time  = result.get("response_time", 0.0),
        chunks_retrieved = len(result.get("chunks", [])),
        escalated      = result.get("escalated", False),
        ticket         = result.get("ticket", ""),
    )

    # update active agent — sidebar highlights this
    st.session_state.active_agent = agent

    # update conversation history
    st.session_state.history.append({"role": "user",      "content": prompt})
    st.session_state.history.append({"role": "assistant", "content": response})

    st.rerun()                    

