# Lease Q&A Assistant

**[Live Demo](https://lease-rag-assistant-rvzqedawxf8eahzqqvvaq8.streamlit.app/)** · **[GitHub Repo](https://github.com/gaganasindhu-ml/lease-rag-assistant)**

A Retrieval-Augmented Generation (RAG) system that answers questions about a residential lease agreement — grounded in the actual lease text, with the exact clause cited for every answer.

---

## The Problem

Lease agreements are dense legal documents. Tenants and property managers need fast, accurate answers to questions like *"How much is the security deposit?"* or *"Can I have a pet?"* — answers about money and legal rights, where a wrong answer matters.

A plain chatbot hallucinates. This system grounds every answer in the retrieved lease text, cites the clause it came from, and refuses to answer when the information isn't in the document.

---

## How It Works

```
residential_lease_agreement.docx
        │
        ▼
  Split into 18 sections by heading style (Heading 1)
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
  LLM (Groq) generates an answer from ONLY the retrieved chunks
        │
        ▼
  Answer + source clause
```

---

## Evaluation

Evaluated against a hand-written 22-question golden set covering direct lookups, scenario questions, paraphrases, a multi-clause question, and three out-of-scope questions that must be refused.

| Metric | Score |
|---|---|
| **Top-1 retrieval accuracy** | **17/19 (89%)** |
| Top-3 retrieval accuracy | 19/19 (100%) |
| Refusal accuracy (out-of-scope) | 3/3 (100%) |

### What each metric means

- **Top-1 retrieval** — the correct section was the *first* result returned.
- **Top-3 retrieval** — the correct section was *somewhere* in the top 3 returned.
- **Refusal** — for the 3 questions the lease does not answer (painting, pest control, move-in inspection), the system said "I don't know" instead of inventing an answer.

### Failure Analysis

Two questions did not return the correct section as the top-1 result:

**1. "I still haven't paid rent after 10 days. What happens now?"**
- Expected: `5. Late Fees`
- Retrieved: `16. Default` (1st), `5. Late Fees` (2nd)
- **Cause:** both sections mention unpaid rent and "10 days". Section 5 covers escalating late fees ($10/day); Section 16 covers default and eviction. The question is genuinely ambiguous in the document's own language — a human would also have to reason about which applies.

**2. "Can I rent out the apartment to someone else?"**
- Expected: `8. Use of Premises`
- Retrieved: `Title` (1st), `4. Rent` (2nd), `8. Use of Premises` (3rd)
- **Cause:** lexical overlap. "Rent out" (subletting) shares the word "rent" with "4. Rent" (rent payment). The embedding matched the shared word, not the meaning.

Both failures come from the same root cause: **the same word appearing in different contexts.** The correct section was still in the top 3 both times, so the LLM could still answer correctly.

### What I'd Improve

1. **Cross-encoder re-ranker** — score each retrieved chunk against the full question, not just embedding similarity, to improve top-1 precision.
2. **Query expansion** — expand "rent out" to "sublet, assign lease" before retrieval.
3. **Hybrid search (BM25 + dense)** — combine keyword matching with semantic matching for exact-term queries.
4. **Section-aware chunking** — keep each section as a single chunk where it fits, so a clause is never split mid-sentence.

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python |
| Document parsing | python-docx |
| Orchestration | LangChain |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store | Qdrant |
| Generation | Groq |
| Frontend | Streamlit |
| Evaluation | Custom (hand-written golden set) |

---

## Project Structure

```
lease-rag-assistant/
├── lease_qa_system.py     # RAG pipeline (LeaseRAG class)
├── streamlit_app.py       # Streamlit web frontend
├── evaluate.py            # evaluation script
├── golden_set.json        # 22 hand-written test questions
├── residential_lease_agreement.docx
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. set your Groq API key
export GROQ_API_KEY="your-key-here"

# 3. ask a question from the command line
python lease_qa_system.py

# 4. run the evaluation
python evaluate.py

# 5. or launch the web app
streamlit run streamlit_app.py
```

---

## Key Design Decisions

**Why split by heading style, not text pattern?**
The document already tags headings (`Heading 1`). Reading the style is more robust than guessing from text like `startswith("Clause")` — it works even if the wording changes.

**Why attach the clause name to every chunk?**
So the answer can cite its source. A chunk without metadata is just text — it can't tell you where it came from.

**Why instruct the LLM to answer only from context?**
To prevent hallucination. When the answer isn't in the retrieved chunks, the system should say so, not make something up. This is tested with 3 out-of-scope questions.

**Why a hand-written golden set instead of an auto-generated one?**
Auto-generating questions from the same chunks the system retrieves creates *circular evaluation* — the test is built from the answer. A hand-written set is an independent test.

**Why validate that the document is a real .docx before parsing?**
A `.docx` is a zip archive. Checking `zipfile.is_zipfile()` first turns a cryptic crash into a clear error message when a file is corrupted, empty, or a Git LFS pointer.

---

## Limitations

- Single-document only (no cross-document reasoning)
- Dense retrieval only (no hybrid search or re-ranking yet)
- Small corpus (18 sections) — accuracy will differ at scale
- No confidence scoring or escalation layer
- English only
