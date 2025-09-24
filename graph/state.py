from typing import List, Dict, TypedDict, Optional

class VideoState(TypedDict, total=False):
    # Inputs
    video_path: str
    user_query: Optional[str]

    # Outputs
    summary: str
    topics: List[str]
    search_results: List[Dict]       # [{topic, results: [{title, url, snippet}]}]
    questions: List[str]             # suggested questions
