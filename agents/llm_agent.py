import os
from dotenv import load_dotenv

from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo

load_dotenv()

def build_agent() -> Agent:
    """
    Single LLM agent used for all reasoning steps (summary, topics, questions).
    We keep DuckDuckGo tool attached in case you want tool-use by prompt,
    but our graph uses `duckduckgo_search` directly for deterministic results.
    """
    # Model id can be swapped (gemini-2.0-flash-exp or gemini-1.5-flash, etc.)
    return Agent(
        name="Video Intelligence Agent",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )
