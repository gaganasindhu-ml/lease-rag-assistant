"""Lease RAG Q&A system.

A Retrieval-Augmented Generation pipeline over a residential lease agreement.
Answers are grounded in the lease text with clause-level citations.
"""

import os

import docx
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path):
    """Find a document path, trying the repo root and a data/ folder.

    Works whether the file sits next to this script, in a data/ folder,
    or is given as an absolute path.
    """
    candidates = [
        path,
        os.path.join(BASE_DIR, path),
        os.path.join(BASE_DIR, "data", os.path.basename(path)),
        os.path.join(BASE_DIR, "data", path),
        os.path.join("data", path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find '%s'. Looked in: %s" % (path, candidates)
    )


class LeaseRAG:
    def __init__(self, doc_path):
        self.doc_path = doc_path
        self.sections = []
        self.chunks = []
        self.vectorstore = None
        self.llm = None
        self.chain = None

    def ingest(self):
        doc = docx.Document(resolve_path(self.doc_path))
        sections = []
        current = None
        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            if "Heading" in p.style.name:
                current = {"heading": text, "lines": []}
                sections.append(current)
            else:
                current["lines"].append(text)
        self.sections = sections
        return sections

    def chunk_sections(self, size=200):
        all_chunks = []
        for section in self.sections:
            text = " ".join(section["lines"])
            for i in range(0, len(text), size):
                piece = text[i:i + size]
                all_chunks.append({"text": piece, "section": section["heading"]})
        self.chunks = all_chunks
        return all_chunks

    def index(self):
        if not self.chunks:
            print("No chunks. Run ingest() and chunk_sections() first.")
            return
        docs = [
            Document(page_content=c["text"], metadata={"section": c["section"]})
            for c in self.chunks
        ]
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name="lease",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        self.vectorstore = QdrantVectorStore(
            client=client,
            collection_name="lease",
            embedding=embeddings,
        )
        self.vectorstore.add_documents(docs)
        print(f"Indexed {len(docs)} chunks")

    def _ensure_llm(self):
        if self.llm is None:
            self.llm = ChatGroq(
                api_key=os.environ.get("bot"),
                model="openai/gpt-oss-20b",
            )
            prompt = ChatPromptTemplate.from_template(
                """Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on this document."

Context:
{context}

Question: {question}

Answer:"""
            )
            self.chain = prompt | self.llm | StrOutputParser()

    def retrieve(self, question, k=3):
        return self.vectorstore.similarity_search(question, k=k)

    def ask(self, question, k=3):
        if self.vectorstore is None:
            print("No vectorstore. Run index() first.")
            return
        self._ensure_llm()
        results = self.retrieve(question, k=k)
        context = "\n\n".join(
            f"[{r.metadata.get('section', 'unknown')}] {r.page_content}"
            for r in results
        )
        return self.chain.invoke({"context": context, "question": question})


if __name__ == "__main__":
    rag = LeaseRAG("residential_lease_agreement.docx")
    rag.ingest()
    rag.chunk_sections(200)
    rag.index()
    print(rag.ask("How much is the security deposit?"))
