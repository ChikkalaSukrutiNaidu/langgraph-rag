from services.pdf_loader import load_pdf
from services.retriever import split_documents
from services.graph_builder import build_graph


docs = load_pdf("data/sample.pdf")

chunks = split_documents(docs)

context = "\n".join(
    [doc.page_content for doc in chunks[:5]]
)

graph = build_graph()

result = graph.invoke(
    {
        "question":
        "What is this document about?",
        "context":
        context
    }
)

print(result["answer"])