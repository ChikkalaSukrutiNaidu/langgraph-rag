from services.pdf_loader import load_pdf
from services.retriever import split_documents
from services.faiss_store import create_vectorstore

docs = load_pdf("data/sample.pdf")

chunks = split_documents(docs)

db = create_vectorstore(chunks)

print("FAISS Created")