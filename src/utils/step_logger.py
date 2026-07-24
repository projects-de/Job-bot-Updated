import os
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_FILE_PATH = str(BASE_DIR / "data" / "application_steps.log")

def log_step(job_id: str, agent: str, step_message: str, level: str = "INFO"):
    """
    Logs a detailed step taken by an agent during the application process.
    The log is appended to a centralized application_steps.log file.
    """
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    timestamp = datetime.utcnow().isoformat()
    log_entry = f"[{timestamp}] [{level}] [Job: {job_id}] [{agent}] {step_message}\n"
    
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Failed to write to step logger: {e}")
