from langgraph.graph import StateGraph, END
from graph.state import VideoState
from graph.nodes import (
    summarize_video_node,
    extract_topics_node,
    web_search_node,
    generate_questions_node,
)

def build_workflow():
    graph = StateGraph(VideoState)

    graph.add_node("summarize", summarize_video_node)
    graph.add_node("topics", extract_topics_node)
    graph.add_node("search", web_search_node)
    graph.add_node("questions", generate_questions_node)

    graph.set_entry_point("summarize")
    graph.add_edge("summarize", "topics")
    graph.add_edge("topics", "search")
    graph.add_edge("search", "questions")
    graph.add_edge("questions", END)

    return graph.compile()
