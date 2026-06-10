from services.pdf_loader import load_pdf
from services.retriever import split_documents

docs = load_pdf("data/sample.pdf")

chunks = split_documents(docs)

print("Chunks:", len(chunks))
print(chunks[0].page_content[:500])