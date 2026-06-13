import streamlit as st

from services.pdf_loader import load_pdf
from services.retriever import split_documents
from services.graph_builder import build_graph


st.title("🏏 IPL LangGraph Assistant")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with open(
        f"data/{uploaded_file.name}",
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    docs = load_pdf(
        f"data/{uploaded_file.name}"
    )

    chunks = split_documents(docs)

    question = st.text_input(
        "Ask a question"
    )

    if question:

        graph = build_graph()

        result = graph.invoke(
            {
                "question": question,
                "chunks": chunks
            }
        )

        st.subheader("Answer")

        st.write(
            result["answer"]
        )