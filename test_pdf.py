from services.pdf_loader import load_pdf

docs = load_pdf("data/sample.pdf")

print("Pages:", len(docs))
print(docs[0].page_content[:500])