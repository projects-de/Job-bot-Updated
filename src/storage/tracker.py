import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = str(BASE_DIR / "data")
UPLOADS_DIR = str(BASE_DIR / "data" / "uploads")
PROFILES_DIR = str(BASE_DIR / "data" / "profiles")
APPLICATIONS_DIR = str(BASE_DIR / "data" / "applications")
TRACKER_FILE = str(BASE_DIR / "data" / "tracker.json")
METADATA_FILE = str(BASE_DIR / "data" / "metadata.json")

_lock = threading.Lock()


class JobStatus:
    DISCOVERED = "DISCOVERED"
    TAILORED = "TAILORED"
    READY = "READY"
    SUBMITTED = "SUBMITTED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"
    SKIPPED = "SKIPPED"


def _ensure_dirs():
    for d in (DATA_DIR, UPLOADS_DIR, PROFILES_DIR, APPLICATIONS_DIR):
        os.makedirs(d, exist_ok=True)


def _read_tracker() -> dict:
    _ensure_dirs()
    if not os.path.exists(TRACKER_FILE):
        return {"applications": []}
    try:
        with open(TRACKER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"applications": []}


def _write_tracker(data: dict):
    _ensure_dirs()
    with open(TRACKER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _read_metadata() -> dict:
    _ensure_dirs()
    if not os.path.exists(METADATA_FILE):
        return {"last_scan_time": None, "groq_api_usage": 0}
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"last_scan_time": None, "groq_api_usage": 0}


def _write_metadata(data: dict):
    _ensure_dirs()
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_last_scan_time():
    with _lock:
        data = _read_metadata()
        data["last_scan_time"] = datetime.now(timezone.utc).isoformat()
        _write_metadata(data)


def increment_api_usage():
    with _lock:
        data = _read_metadata()
        data["groq_api_usage"] = data.get("groq_api_usage", 0) + 1
        _write_metadata(data)


def get_metadata() -> dict:
    with _lock:
        return _read_metadata()


def log_application(job_id: str, job_title: str, company: str, status: str, reason: str = None):
    with _lock:
        data = _read_tracker()
        apps = data.get("applications", [])
        now = datetime.now(timezone.utc).isoformat()

        for entry in apps:
            if entry.get("job_id") == job_id:
                entry["status"] = status
                entry["updated_at"] = now
                if reason:
                    entry["reason"] = reason
                _write_tracker(data)
                return

        apps.append({
            "job_id": job_id,
            "job_title": job_title,
            "company": company,
            "status": status,
            "reason": reason,
            "created_at": now,
            "updated_at": now,
        })
        data["applications"] = apps
        _write_tracker(data)


def get_tracker_summary() -> dict:
    with _lock:
        return _read_tracker()


def save_parsed_profile(profile: dict):
    _ensure_dirs()
    path = os.path.join(PROFILES_DIR, "parsed_profile.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def save_tailored_application(job_id: str, payload: dict):
    _ensure_dirs()
    job_dir = os.path.join(APPLICATIONS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    path = os.path.join(job_dir, "tailored_cv.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_jobs(jobs: list, suffix: str = "all"):
    _ensure_dirs()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(DATA_DIR, f"jobs-{suffix}-{timestamp}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)


def save_summary(summary: dict):
    _ensure_dirs()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(DATA_DIR, f"summary-{timestamp}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def get_dashboard_stats() -> dict:
    with _lock:
        data = _read_tracker()
        meta = _read_metadata()
    apps = data.get("applications", [])

    counts = {
        JobStatus.DISCOVERED: 0,
        JobStatus.TAILORED: 0,
        JobStatus.READY: 0,
        JobStatus.SUBMITTED: 0,
        JobStatus.FAILED: 0,
        JobStatus.EXPIRED: 0,
        JobStatus.SKIPPED: 0,
    }
    for entry in apps:
        s = entry.get("status")
        if s in counts:
            counts[s] += 1

    return {
        "total": len(apps),
        "discovered": counts[JobStatus.DISCOVERED],
        # A skipped job still went through tailoring (CV/cover letter were generated for
        # it before the user reviewed and skipped it in-browser), so it counts toward the
        # "Tailored" stat on the dashboard and appears in the Tailored Pipeline list too.
        "tailored": counts[JobStatus.TAILORED] + counts[JobStatus.SKIPPED],
        "ready": counts[JobStatus.READY],
        "submitted": counts[JobStatus.SUBMITTED],
        "failed": counts[JobStatus.FAILED],
        "expired": counts[JobStatus.EXPIRED],
        "skipped": counts[JobStatus.SKIPPED],
        "last_scan_time": meta.get("last_scan_time"),
        "groq_api_usage": meta.get("groq_api_usage", 0)
    }
