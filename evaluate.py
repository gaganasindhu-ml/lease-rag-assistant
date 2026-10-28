"""Evaluation script for the Lease Q&A Assistant.

Loads the golden set, runs every question through the pipeline, and reports
retrieval accuracy (did it find the right section?) and refusal accuracy
(did it refuse questions the lease does not answer?).

Usage:
    python evaluate.py
"""

import json

from lease_qa_system import LeaseRAG

DOC_PATH = "residential_lease_agreement.docx"
GOLDEN_PATH = "golden_set.json"

REFUSAL_MARKERS = [
    "don't know",
    "do not know",
    "not address",
    "does not address",
    "not mentioned",
    "no information",
    "not specify",
    "does not specify",
    "not covered",
    "doesn't mention",
]


def evaluate(rag, questions):
    top1 = top3 = answerable = 0
    refusals = refusal_total = 0
    misses = []

    for item in questions:
        question = item["question"]

        if item["answerable"]:
            answerable += 1
            results = rag.vectorstore.similarity_search(question, k=3)
            retrieved = [r.metadata["section"] for r in results]
            expected = item["expected_section"]

            if retrieved and retrieved[0] == expected:
                top1 += 1
            if expected in retrieved:
                top3 += 1
            else:
                misses.append({
                    "id": item["id"],
                    "question": question,
                    "expected": expected,
                    "got": retrieved,
                })
        else:
            refusal_total += 1
            answer = rag.ask(question).lower()
            if any(marker in answer for marker in REFUSAL_MARKERS):
                refusals += 1
            else:
                misses.append({
                    "id": item["id"],
                    "question": question,
                    "expected": "REFUSAL",
                    "got": answer,
                })

    return {
        "top1": top1,
        "top3": top3,
        "answerable": answerable,
        "refusals": refusals,
        "refusal_total": refusal_total,
        "misses": misses,
    }


def main():
    rag = LeaseRAG(DOC_PATH)
    rag.ingest()
    rag.chunk_sections(200)
    rag.index()

    golden = json.load(open(GOLDEN_PATH))
    questions = golden["questions"]

    result = evaluate(rag, questions)

    a = result["answerable"]
    r = result["refusal_total"]

    print(f"Top-1 retrieval: {result['top1']}/{a} = {result['top1'] / a:.0%}")
    print(f"Top-3 retrieval: {result['top3']}/{a} = {result['top3'] / a:.0%}")
    print(f"Refusal:         {result['refusals']}/{r} = {result['refusals'] / r:.0%}")
    print()

    if result["misses"]:
        print("MISSES:")
        for miss in result["misses"]:
            print(miss)
    else:
        print("No misses.")


if __name__ == "__main__":
    main()
