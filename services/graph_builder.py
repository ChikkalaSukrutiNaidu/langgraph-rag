from typing import TypedDict
from langgraph.graph import StateGraph, END

from services.llm import llm


class GraphState(TypedDict):
    question: str
    context: str
    answer: str


def retrieve(state):

    return {
        "context": state["context"]
    }


def generate(state):

    prompt = f"""
    Context:
    {state['context']}

    Question:
    {state['question']}

    Answer from context only.
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def build_graph():

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("retrieve")

    workflow.add_edge(
        "retrieve",
        "generate"
    )

    workflow.add_edge(
        "generate",
        END
    )

    return workflow.compile()