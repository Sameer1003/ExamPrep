import json
from typing import List

def try_parse_topics(text: str, fallback_k: int = 6) -> List[str]:
    """
    Try parsing a JSON: {"topics": ["a","b",...]}
    Fallback: extract bullet lines as topics.
    """
    text = text.strip()
    # attempt JSON extraction
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "topics" in data and isinstance(data["topics"], list):
            return [t.strip() for t in data["topics"] if isinstance(t, str) and t.strip()]
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass

    # fallback: split by lines or bullets
    lines = [l.strip("-•* ").strip() for l in text.splitlines() if l.strip()]
    # Keep short-ish phrases, de-duplicate
    uniq = []
    for l in lines:
        if l and l not in uniq and len(l) <= 80:
            uniq.append(l)
    if not uniq:
        return lines[:fallback_k]
    return uniq[:max(fallback_k, len(uniq))]
