from services.pdf_loader import load_pdf
from services.retriever import (
    split_documents,
    simple_retrieve
)
from services.graph_builder import build_graph


docs = load_pdf("data/sample.pdf")

chunks = split_documents(docs)

question = "What is this document about?"

relevant_docs = simple_retrieve(
    chunks,
    question
)

context = "\n".join(
    [doc.page_content for doc in relevant_docs]
)

graph = build_graph()

result = graph.invoke(
    {
        "question": question,
        "context": context
    }
)

print(result["answer"])