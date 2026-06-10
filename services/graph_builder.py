from typing import TypedDict

from langgraph.graph import StateGraph, END

from services.llm import llm


class GraphState(TypedDict):
    question: str
    context: str
    answer: str


def retrieve_node(state):

    question = state["question"]

    context = (
        "This document contains IPL team profiles, "
        "player statistics, venue data, match predictions "
        "and Dream11 related information."
    )

    return {
        "question": question,
        "context": context
    }


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


graph = StateGraph(GraphState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "generate")

graph.add_edge("generate", END)

app = graph.compile()