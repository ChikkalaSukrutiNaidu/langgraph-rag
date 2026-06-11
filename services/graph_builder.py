from typing import TypedDict

from langgraph.graph import StateGraph, END

from services.llm import llm
from services.retriever import simple_retrieve


class GraphState(TypedDict):
    question: str
    chunks: list
    context: str
    answer: str


def retrieve_node(state):

    print("Retrieving context...")

    question = state["question"]
    chunks = state["chunks"]

    docs = simple_retrieve(
        chunks,
        question
    )

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    return {
        "context": context
    }


def generate_node(state):

    print("Generating answer...")

    question = state["question"]
    context = state["context"]

    prompt = f"""
    Answer using only the context.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def build_graph():

    graph = StateGraph(GraphState)

    graph.add_node(
        "retrieve",
        retrieve_node
    )

    graph.add_node(
        "generate",
        generate_node
    )

    graph.set_entry_point(
        "retrieve"
    )

    graph.add_edge(
        "retrieve",
        "generate"
    )

    graph.add_edge(
        "generate",
        END
    )

    return graph.compile()