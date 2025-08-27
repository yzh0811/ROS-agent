from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


def create_graph(llm) -> StateGraph:
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    try:
        graph_builder = StateGraph(State)

        def chatbot(state: State) -> dict:
            return {"messages": [llm.invoke(state["messages"]) ]}

        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        return graph_builder.compile()
    except Exception as e:
        raise RuntimeError(f"Failed to create graph: {str(e)}") 