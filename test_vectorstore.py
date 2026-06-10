from services.pdf_loader import load_pdf
from services.retriever import (
    split_documents,
    create_vectorstore
)

docs = load_pdf(
    "data/sample.pdf"
)

chunks = split_documents(docs)

vectorstore = create_vectorstore(chunks)

print("Vector DB Created Successfully")