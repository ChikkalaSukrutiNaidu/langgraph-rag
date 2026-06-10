import streamlit as st

from services.pdf_loader import load_pdf
from services.retriever import (
    split_documents,
    simple_retrieve
)
from services.graph_builder import build_graph


st.title("🏏 IPL LangGraph Assistant")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    with open(
        "data/uploaded.pdf",
        "wb"
    ) as f:
        f.write(uploaded_file.getbuffer())

    docs = load_pdf(
        "data/uploaded.pdf"
    )

    chunks = split_documents(docs)

    question = st.text_input(
        "Ask a question"
    )

    if question:

        relevant_docs = simple_retrieve(
            chunks,
            question
        )

        context = "\n".join(
            [
                doc.page_content
                for doc in relevant_docs
            ]
        )

        graph = build_graph()

        result = graph.invoke(
            {
                "question": question,
                "context": context
            }
        )

        st.write("### Answer")

        st.write(
            result["answer"]
        )