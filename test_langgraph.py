from services.pdf_loader import load_pdf
from services.retriever import split_documents
from services.graph_builder import build_graph

docs = load_pdf("data/sample.pdf")

chunks = split_documents(docs)

graph = build_graph()

question = input("Ask Question: ")

result = graph.invoke(
    {
        "question": question,
        "chunks": chunks
    }
)

print("\nAnswer:\n")
print(result["answer"])