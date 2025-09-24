import os
import json
import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "agent.log")


def log_response(step: str, content):
    """
    Append one step's output to a global text log file.
    """
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n=== {step.upper()} @ {datetime.datetime.now().isoformat()} ===\n")
        if isinstance(content, (dict, list)):
            f.write(json.dumps(content, ensure_ascii=False, indent=2))
        else:
            f.write(str(content))
        f.write("\n")


def save_run_log(state: dict) -> str:
    """
    Save the full pipeline run to a JSON file.
    Each run gets a unique timestamped file.
    """
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(LOG_DIR, f"run_{run_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return filepath
