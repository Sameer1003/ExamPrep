import time
import json
from typing import Dict, Any, List

import google.generativeai as genai
from google.generativeai import upload_file, get_file

from duckduckgo_search import DDGS

from agents.llm_agent import build_agent
from utils.parsing import try_parse_topics
from graph.state import VideoState
from utils.logger import log_response


# Build one agent instance for all nodes
AGENT = build_agent()


# ============== Node 1: Summarize Video =================
def summarize_video_node(state: VideoState) -> VideoState:
    video_path = state.get("video_path")
    if not video_path:
        raise ValueError("summarize_video_node: 'video_path' is missing in state.")

    # Upload video to Gemini and poll
    processed_video = upload_file(video_path)
    while processed_video.state.name == "PROCESSING":
        time.sleep(1)
        processed_video = get_file(processed_video.name)

    prompt = (
        "You are an expert audiovisual analyst. Summarize the video clearly for a general audience.\n"
        "- Use concise bullets with headings for sections.\n"
        "- Include key facts, numbers, claims, and takeaways.\n"
        "- Keep it under ~200-300 words.\n"
        "- Output in markdown."
    )

    resp = AGENT.run(prompt, videos=[processed_video])
    state["summary"] = resp.content.strip()
    log_response("summary", state["summary"])
    return state


# ============== Node 2: Extract Topics ===================
def extract_topics_node(state: VideoState) -> VideoState:
    """
    Extract main topics from the summary.
    """
    prompt = f"""
    From this summary, list **3-6 main topics** as plain text bullet points.
    Do not use JSON, code blocks, or formatting. Only output a clean list.

    Summary:
    {state['summary']}
    """

    resp = AGENT.run(prompt)
    topics_raw = resp.content.strip().split("\n")

    # Clean lines like "- Topic" → "Topic"
    cleaned = [t.strip("-• ").strip() for t in topics_raw if t.strip()]
    state["topics"] = cleaned
    log_response("topics", state["topics"])
    return state


# ============== Node 3: Web Search =======================
def web_search_node(state: VideoState) -> VideoState:
    topics = state.get("topics", [])
    if not topics:
        raise ValueError("web_search_node: 'topics' missing or empty in state.")

    results: List[Dict[str, Any]] = []

    with DDGS() as ddgs:
        for t in topics:
            hits = ddgs.text(t, max_results=5, safesearch="moderate")
            topic_results = []
            for h in hits:
                topic_results.append({
                    "title": h.get("title"),
                    "url": h.get("href") or h.get("url"),
                    "snippet": h.get("body") or h.get("snippet")
                })
            results.append({"topic": t, "results": topic_results})

    state["search_results"] = results
    log_response("search_results", state["search_results"])
    return state


# ============== Node 4: Generate Questions ===============
def generate_questions_node(state: VideoState) -> VideoState:
    topics = state.get("topics", [])
    search_results = state.get("search_results", [])
    user_query = state.get("user_query", "") or ""

    # Keep a compact payload for the LLM
    brief_payload = []
    for pack in search_results:
        t = pack["topic"]
        rs = pack["results"][3:]  # top 3 per topic for brevity
        brief_payload.append({
            "topic": t,
            "results": [{"title": r["title"], "url": r["url"], "snippet": r["snippet"]} for r in rs]
        })

    prompt = f"""
You are a helpful research assistant. Based on the following topics and brief web snippets,
generate 8–12 insightful, open-ended questions that a curious user should ask to dive deeper.
Questions should be specific, non-trivial, and avoid yes/no framing. If a user query is provided, bias questions toward it.

Return STRICT JSON list like:
["Question 1", "Question 2", "..."]

User query (optional): {user_query}

Topics:
{json.dumps(topics, ensure_ascii=False, indent=2)}

Brief web context:
{json.dumps(brief_payload, ensure_ascii=False, indent=2)}
"""
    resp = AGENT.run(prompt)

    # Parse back to list
    questions: List[str] = []
    try:
        data = json.loads(resp.content)
        if isinstance(data, list):
            questions = [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        # fallback: split lines
        for line in resp.content.splitlines():
            line = line.strip("-•* ").strip()
            if line:
                questions.append(line)

    state["questions"] = questions[:12]
    log_response("questions", state["questions"])
    return state
