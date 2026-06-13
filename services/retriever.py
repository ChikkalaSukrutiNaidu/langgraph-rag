from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(documents)


def simple_retrieve(chunks, query, k=4):

    query = query.lower()

    scored_chunks = []

    for chunk in chunks:

        text = chunk.page_content.lower()

        score = 0

        for word in query.split():
            if word in text:
                score += 1

        scored_chunks.append((score, chunk))

    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [chunk for score, chunk in scored_chunks[:k]]