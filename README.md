# 🛍️ Multi-Agent Customer Support System
### Built for ShopSmart E-Commerce — Powered by LangGraph, FAISS, and Groq

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0.10-orange?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-purple?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-3.10.0-blue?style=flat-square&logo=mlflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55.0-red?style=flat-square&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker)

---

## What This Is

A production-grade multi-agent AI system that handles customer support for ShopSmart e-commerce. Every customer message is classified by intent and sentiment, then routed to the right specialized agent — automatically.

Not a single chatbot. Four agents working together.

---

## Live Demo

> Try the App Here:  https://multi-agent-customer-support-system-mahesh.streamlit.app/

Please follow or copy & paste link your browser and Wake up the App if it is sleeping !!

---

## Visual Demo
Knowledge Agent in force:

<img width="1919" height="935" alt="Screenshot 2026-03-10 191741" src="https://github.com/user-attachments/assets/ae0e47c3-3ab8-4a39-8db6-13adc7197fed" />


Action Agent in force:

<img width="1911" height="941" alt="Screenshot 2026-03-10 191708" src="https://github.com/user-attachments/assets/ef7417f8-d56a-49fd-a6f8-d584bbd6a2f8" />


Escalation Agent in force:

<img width="1903" height="948" alt="Screenshot 2026-03-10 191635" src="https://github.com/user-attachments/assets/2f918c18-fdcd-4954-9268-287b77343b06" />

Metrics:

<img width="1858" height="921" alt="Screenshot 2026-03-10 105333" src="https://github.com/user-attachments/assets/200ebdc0-ab67-43c0-9e3b-9888fa4c1cde" />
<img width="1861" height="930" alt="Screenshot 2026-03-10 105045" src="https://github.com/user-attachments/assets/22c76da7-7909-4c55-9904-41f10b0dd875" />

<img width="1864" height="968" alt="Screenshot 2026-03-10 105017" src="https://github.com/user-attachments/assets/5ea1a14b-2b31-43c5-a9fa-47f8292374e8" />

## How It Works

```
Customer Message
       ↓
 [Classifier Agent]
  detects intent (11 categories)
  detects sentiment (4 levels)
       ↓
  [Router]
  HIGH_DISTRESS → Escalation
  return/refund/order/payment → Action
  everything else → Knowledge
       ↓
┌──────────────┬──────────────┬──────────────┐
│  Knowledge   │    Action    │  Escalation  │
│    Agent     │    Agent     │    Agent     │
│              │              │              │
│ RAG answers  │ Processes    │ Human handoff│
│ from 21 docs │ returns,     │ for distress │
│ via FAISS    │ refunds,     │ and legal    │
│              │ orders       │ threats      │
└──────────────┴──────────────┴──────────────┘
```

---

## Agents

### 🧠 Knowledge Agent
Answers customer questions using RAG (Retrieval Augmented Generation) over 21 ShopSmart documents. Retrieves top-5 relevant chunks from FAISS vector store and generates grounded answers with source citations.

### ⚙️ Action Agent
Handles requests that require processing — returns, refunds, order status, payment issues. Simulates system actions and generates unique ticket references (`SS-XXXXXX`) for tracking.

### 🚨 Escalation Agent
Triggered automatically when sentiment is `HIGH_DISTRESS` or intent is `escalation`. Responds with genuine empathy and connects the customer to a senior specialist with legally mandated 48-hour SLA.

### 🔍 Classifier Agent
Routes every message to the right agent. Classifies into 11 intents and 4 sentiment levels using Groq's `llama-3.3-70b-versatile` at `temperature=0.0` for deterministic routing.

---

## Knowledge Base

21 documents across 4 file types — all processed, chunked, and indexed into FAISS:

| Type | Files | Content |
|------|-------|---------|
| PDF | 6 | Return policy, Shipping policy, Warranty, Privacy, T&C, Payment security |
| DOCX | 6 | Escalation guide, Order SOP, Refund SOP, Damaged goods, Fraud detection, VIP handling |
| CSV | 6 | Product catalog (88 products), FAQ database (100 FAQs), Order codes, Shipping zones, Coupons, Vendors |
| TXT | 3 | Company overview, Agent scripts, Prohibited responses |

---

## Performance

Benchmarked across **63 real customer queries** covering all intents and agent types:

| Metric | Value |
|--------|-------|
| Total queries evaluated | 63 |
| Average response time | ~130ms |
| Vector chunks indexed | 396 |
| Embedding dimensions | 384 |
| Intents supported | 11 |
| Experiment runs logged | 63 |

All 63 runs tracked and visible in MLflow dashboard with full params, metrics, and conversation artifacts.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | LangGraph 1.0.10 |
| LLM | Groq — llama-3.3-70b-versatile |
| Vector Store | FAISS (396 chunks, 384-dim) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Experiment Tracking | MLflow 3.10.0 |
| UI | Streamlit 1.55.0 |
| Containerization | Docker + Docker Compose |
| Document Processing | pypdf, python-docx, pandas |

---

## Project Structure

```
Multi-Agent-Customer-Support-System/
│
├── agents/
│   ├── __init__.py
│   ├── classifier.py       # intent + sentiment detection
│   ├── knowledge.py        # RAG answers from FAISS
│   ├── actions.py          # return/refund/order processing
│   └── escalations.py      # human handoff for distress
│
├── knowledge_base/         # 21 source documents
│   ├── *.pdf               # 6 policy PDFs
│   ├── *.docx              # 6 SOP documents
│   ├── *.csv               # 6 data files
│   └── *.txt               # 3 guideline files
│
├── vectorstore/            # FAISS index (gitignored)
│   ├── index.faiss
│   └── metadata.pkl
│
├── for_agents_core.py      # shared utilities: load, retrieve, generate
├── graph.py                # LangGraph orchestration + routing
├── data_processing_ingestion.py  # document loader + FAISS builder
├── app.py                  # Streamlit chat UI
├── eval.py                 # MLflow batch evaluation
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Intents Supported

```
order_status      → tracking, delivery date, order not received
return_request    → want to return, exchange, pickup
refund_status     → refund not received, refund delay
product_query     → specs, stock, compatibility, brands
payment_issue     → payment failed, charged twice, EMI, COD
complaint         → damaged item, wrong item, missing item
shipping_query    → delivery time, shipping cost, pin code
warranty_claim    → warranty, repair, service center
account_issue     → login, suspended, password reset
escalation        → legal threats, consumer court, social media
general           → greetings, feedback, anything else
```

---

## Sentiment Detection

```
POSITIVE      → happy, satisfied customer
NEUTRAL       → routine query, no emotion
NEGATIVE      → frustrated, dissatisfied
HIGH_DISTRESS → angry, threatening, legal action
               → always triggers Escalation Agent
```

---

## Quickstart

### 1. Clone
```bash
git clone https://github.com/themaheswar1/Multi-Agent-Customer-Support-System.git
cd Multi-Agent-Customer-Support-System
```

### 2. Setup environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add API key
```bash
cp .env.example .env
# add your GROQ_API_KEY to .env
```

### 4. Build vector store
```bash
python data_processing_ingestion.py
```

### 5. Run app
```bash
streamlit run app.py
```

### 6. Run batch evaluation
```bash
python eval.py
mlflow ui  # view results at http://localhost:5000
```

---

## Docker

```bash
# build and run
docker compose up --build

# app available at http://localhost:8501
```

---

## Environment Variables

```bash
# .env.example
GROQ_API_KEY=your_groq_api_key_here
```

---

## Git Branching Strategy

```
master          → production, always stable
dev             → integration branch
feature/xxx     → individual feature branches

feature/* → dev → master
```

---

## MLflow Tracking

Every conversation turn and batch evaluation run is logged to MLflow:

```
Params logged:   intent, sentiment, agent, escalated, query_length
Metrics logged:  response_time_sec, response_length, chunks_retrieved
Artifacts:       full conversation text per run
Tags:            agent_type, sentiment, intent, version
```

View dashboard:
```bash
mlflow ui
# http://localhost:5000
```

---

## Roadmap

- [ ] Add response time tracking per agent in app.py
- [ ] Deploy to Streamlit Cloud
- [ ] Add order number validation in Action Agent
- [ ] Add multi-language support
- [ ] Add conversation export feature

---

## Author

**Mahesh** — [@themaheswar1](https://github.com/themaheswar1)


---

Failed*30x, Built it Finally !! 
