from typing import TypedDict

from langgraph.graph import StateGraph, END
from services.llm import llm


class GraphState(TypedDict):
    question: str
    context: str
    answer: str


def generate_node(state):

    question = state["question"]
    context = state["context"]

    prompt = f"""
    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def build_graph():

    graph = StateGraph(GraphState)

    graph.add_node(
        "generate",
        generate_node
    )

    graph.set_entry_point(
        "generate"
    )

    graph.add_edge(
        "generate",
        END
    )

    return graph.compile()