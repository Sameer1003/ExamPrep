import os
import time
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

from graph.workflow import build_workflow
from graph.state import VideoState

from utils.logger import save_run_log

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

st.set_page_config(page_title="Agentic Video Research Assistant", layout="wide")

st.title("🎬 Agentic Video Research Assistant")
st.caption("Summarize a video → extract topics → search the web → propose deep-dive questions.")

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    st.info("Model: Gemini 2.0 Flash (via phidata). Web search: DuckDuckGo.")
    st.markdown("---")
    st.markdown("**Flow**: Summary → Topics → Search → Questions")

# File uploader
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
user_query = st.text_input(
    "Optional: what are you trying to learn?",
    placeholder="e.g., Focus on policy implications and real-world case studies."
)

run_button = st.button("🚀 Run Agentic Pipeline")

if run_button:
    if not video_file:
        st.warning("Please upload a video to proceed.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path, format="video/mp4")

    try:
        with st.spinner("Building graph and analyzing..."):
            workflow = build_workflow()

            # initialize state
            state: VideoState = {
                "video_path": video_path,
                "user_query": user_query.strip() if user_query else None,
            }

            # Run compiled graph
            final_state: VideoState = workflow.invoke(state)  # .invoke for single pass

        # Display results
        st.success("Done!")

        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("📌 Video Summary")
            st.markdown(final_state.get("summary", "_No summary_"))

            st.subheader("🧭 Key Topics")
            topics = final_state.get("topics", [])
            if topics:
                st.markdown("\n".join([f"- {t}" for t in topics]))
            else:
                st.write("_No topics extracted._")

        with col2:
            st.subheader("🤖 Suggested Questions")
            questions = final_state.get("questions", [])
            if questions:
                st.markdown("\n".join([f"- {q}" for q in questions]))
            else:
                st.write("_No questions generated._")

        st.subheader("🔎 Web Results (by topic)")
        sr = final_state.get("search_results", [])
        if not sr:
            st.write("_No search results._")
        else:
            for pack in sr:
                st.markdown(f"### **{pack['topic']}**")
                if not pack["results"]:
                    st.write("_No hits._")
                    continue
                for r in pack["results"]:
                    title = r.get("title") or "Untitled"
                    url = r.get("url") or "#"
                    snippet = r.get("snippet") or ""
                    st.markdown(f"- **[{title}]({url})**  \n  {snippet}")

        final_state: VideoState = workflow.invoke(state)

        # Save full run log
        log_path = save_run_log(final_state)
        st.info(f"Responses logged to: {log_path}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        Path(video_path).unlink(missing_ok=True)

