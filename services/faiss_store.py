from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings


def create_vectorstore(chunks):

    embeddings = FakeEmbeddings(
        size=384
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore