Lease Q&A Assistant
A Retrieval-Augmented Generation (RAG) system that answers questions about a residential lease agreement — grounded in the actual lease text, with the exact clause cited for every answer.

The Problem
Lease agreements are dense legal documents. Tenants and property managers need fast, accurate answers to questions like "How much is the security deposit?" or "Can I have a pet?" — answers about money and legal rights, where a wrong answer matters.

A plain chatbot hallucinates. This system grounds every answer in the retrieved lease text, cites the clause it came from, and refuses to answer when the information isn't in the document.

How It Works


residential_lease_agreement.docx
        │
        ▼
  Split into sections by heading style (Heading 1 / Heading 2)
        │
        ▼
  Chunk each section, attaching the clause name as metadata
        │
        ▼
  Embed each chunk → 384-dim vector (all-MiniLM-L6-v2)
        │
        ▼
  Store vectors + metadata in Qdrant
        │
        ▼  (on a question)
  Embed the question → find the closest chunk vectors
        │
        ▼
  LLM (Groq LLaMA) generates an answer from ONLY the retrieved chunks
        │
        ▼
  Answer + source clause
Evaluation
Evaluated against a hand-written 18-question golden set covering direct lookups, scenario questions, paraphrases, a multi-clause question, and three out-of-scope questions that should be refused.




Metric	Score
Top-1 retrieval accuracy	14/15 (93%)
Top-3 retrieval accuracy	15/15 (100%)
Refusal accuracy (out-of-scope)	3/3 (100%)
Failure Analysis
The single top-1 miss was:

"Can I rent out my spare room to a friend?" Expected: Clause 6: Subletting Retrieved: Clause 2: Rent and Due Date (top-1), Clause 6 at position 3

Cause: lexical overlap. The phrase "rent out" shares the word "rent" with "Rent and Due Date", so the embedding model matched the wrong clause. "Rent out" (subletting) and "rent payment" are different concepts that share a word. The correct clause was still in the top 3, so the LLM could still answer correctly.

What I'd Improve
Cross-encoder re-ranker — look at the full query, not just word overlap, to improve top-1 precision.
Query expansion — expand "rent out" to "sublet, assign lease" before retrieval.
Hybrid search (BM25 + dense) — combine keyword matching with semantic matching for exact-term queries.
Tech Stack



Component	Tool
Language	Python
Document parsing	python-docx
Orchestration	LangChain
Embeddings	sentence-transformers (all-MiniLM-L6-v2)
Vector store	Qdrant
Generation	Groq (LLaMA)
Evaluation	Custom (hand-written golden set)
Project Structure


lease-rag-assistant/
├── lease_qa_system.py    # full RAG pipeline (LeaseRAG class)
├── data/
│   └── residential_lease_agreement.docx
├── eval/
│   └── golden_set.json
├── requirements.txt
├── .gitignore
└── README.md
How to Run
bash


# 1. install dependencies
pip install -r requirements.txt

# 2. set your Groq API key
export GROQ_API_KEY="your-key-here"

# 3. build the index
python -m src.ingest
python -m src.index

# 4. ask a question
python -m src.generate "How much is the security deposit?"

# 5. run the evaluation
python evaluate.py
Key Design Decisions
Why split by heading style, not text pattern? The document already tags headings (Heading 1, Heading 2). Reading the style is more robust than guessing from text like startswith("Clause") — it works even if the wording changes.

Why attach the clause name to every chunk? So the answer can cite its source. A chunk without metadata is just text — it can't tell you where it came from.

Why instruct the LLM to answer only from context? To prevent hallucination. When the answer isn't in the retrieved chunks, the system should say so, not make something up. This is tested with 3 out-of-scope questions.

Why a hand-written golden set instead of an auto-generated one? Auto-generating questions from the same chunks the system retrieves creates circular evaluation — the test is built from the answer. A hand-written set is an independent test.

Limitations
Single-document only (no cross-document reasoning)
Dense retrieval only (no hybrid search yet)
Small corpus (9 chunks) — accuracy will differ at scale
No confidence scoring / escalation layer
