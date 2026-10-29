"""Streamlit frontend for the Lease Q&A Assistant."""

import os

import streamlit as st

# LeaseRAG (ChatGroq) reads GROQ_API_KEY, so map the Streamlit secret "bot" to it
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

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


# Show the real error on screen instead of Streamlit's redacted message
try:
    rag = load_rag()
except Exception as e:
    st.error(f"Failed to load the lease: {e}")
    st.stop()

question = st.text_input(
    "Your question:",
    placeholder="How much is the security deposit?",
)

if st.button("Ask") and question:
    with st.spinner("Searching the lease..."):
        try:
            answer = rag.ask(question)
        except Exception as e:
            st.error(f"Could not get an answer: {e}")
            st.stop()
    st.success(answer)

    with st.expander("Retrieved clauses"):
        for r in rag.retrieve(question, k=3):
            st.write(f"**{r.metadata.get('section', 'unknown')}**")
            st.write(r.page_content)
