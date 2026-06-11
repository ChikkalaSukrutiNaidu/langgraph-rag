from typing import TypedDict

from langgraph.graph import StateGraph, END

from services.llm import llm
from services.retriever import simple_retrieve


class GraphState(TypedDict):
    question: str
    chunks: list
    route: str
    context: str
    answer: str


def router_node(state):

    question = state["question"].lower()

    if any(word in question for word in
           ["captain", "team", "csk", "rcb", "mi"]):

        route = "team"

    elif any(word in question for word in
             ["player", "runs", "wickets",
              "kohli", "dhoni", "gaikwad"]):

        route = "player"

    else:
        route = "general"

    print("Route:", route)

    return {"route": route}


def team_node(state):

    print("Team Agent Running")

    docs = simple_retrieve(
        state["chunks"],
        state["question"]
    )

    context = "\n".join(
        [d.page_content for d in docs]
    )

    return {"context": context}


def player_node(state):

    print("Player Agent Running")

    docs = simple_retrieve(
        state["chunks"],
        state["question"]
    )

    context = "\n".join(
        [d.page_content for d in docs]
    )

    return {"context": context}


def general_node(state):

    print("General Agent Running")

    docs = simple_retrieve(
        state["chunks"],
        state["question"]
    )

    context = "\n".join(
        [d.page_content for d in docs]
    )

    return {"context": context}


def generate_node(state):

    prompt = f"""
    Context:
    {state['context']}

    Question:
    {state['question']}

    Answer:
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def route_decision(state):

    return state["route"]


def build_graph():

    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)

    graph.add_node("team", team_node)

    graph.add_node("player", player_node)

    graph.add_node("general", general_node)

    graph.add_node("generate", generate_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "team": "team",
            "player": "player",
            "general": "general"
        }
    )

    graph.add_edge("team", "generate")

    graph.add_edge("player", "generate")

    graph.add_edge("general", "generate")

    graph.add_edge("generate", END)

    return graph.compile()