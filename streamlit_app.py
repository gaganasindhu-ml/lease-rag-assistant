"""Streamlit frontend for the Lease Q&A Assistant."""

import os

import streamlit as st

# Make the Streamlit secret available as an env var for LeaseRAG
os.environ["bot"] = st.secrets["bot"]

from lease_qa_system import LeaseRAG  # noqa: E402

st.set_page_config(page_title="Lease Q&A Assistant", page_icon="📄")

st.title("📄 Lease Q&A Assistant")
st.write(
    "Ask questions about the residential lease agreement. "
    "Answers are grounded in the lease text, with the clause cited."
)


@st.cache_resource
def load_rag():
    rag = LeaseRAG("residential_lease_agreement.docx")
    rag.ingest()
    rag.chunk_sections(200)
    rag.index()
    return rag


rag = load_rag()

question = st.text_input(
    "Your question:",
    placeholder="How much is the security deposit?",
)

if st.button("Ask") and question:
    with st.spinner("Searching the lease..."):
        answer = rag.ask(question)
    st.success(answer)

    with st.expander("Retrieved clauses"):
        for r in rag.retrieve(question, k=3):
            st.write(f"**{r.metadata.get('section', 'unknown')}**")
            st.write(r.page_content)
