import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Same fixed-offset Pakistan Standard Time used by the daily report
# (src/workflow/daily_report.py) — kept in sync so the dashboard's "today" and
# the emailed report's "today" always agree on the same calendar-day boundary.
PKT = timezone(timedelta(hours=5))

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


def _pkt_date(iso_string: str):
    """PKT calendar date of an ISO timestamp, or None if unparseable/empty."""
    if not iso_string:
        return None
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(PKT).date()
    except ValueError:
        return None


def is_tailored_today(created_at: str) -> bool:
    """
    Whether a job's tracker entry was first tailored on the current PKT
    calendar day. A job's `created_at` is written exactly once, at the moment
    it's first tailored (DISCOVERED is never persisted to tracker.json — it's
    only a UI default for jobs with no entry yet), so `created_at` doubles as
    "tailored_at" without needing a separate field.
    """
    entry_date = _pkt_date(created_at)
    return entry_date is not None and entry_date == datetime.now(PKT).date()


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
    tailored_today = 0
    for entry in apps:
        s = entry.get("status")
        if s in counts:
            counts[s] += 1
        # Tailored/Ready/Skipped all originate from the same TAILORED write (see
        # is_tailored_today), so created_at is that job's tailoring date regardless
        # of which of the three statuses it's since moved to.
        if s in (JobStatus.TAILORED, JobStatus.READY, JobStatus.SKIPPED) and is_tailored_today(entry.get("created_at")):
            tailored_today += 1

    return {
        "total": len(apps),
        "discovered": counts[JobStatus.DISCOVERED],
        # A skipped job still went through tailoring (CV/cover letter were generated for
        # it before the user reviewed and skipped it in-browser), so it counts toward the
        # "Tailored" stat on the dashboard and appears in the Tailored Pipeline list too.
        # This is the all-time total (kept for anything else that relies on it) —
        # "tailored_today" below is the day-scoped count the live dashboard card uses.
        "tailored": counts[JobStatus.TAILORED] + counts[JobStatus.SKIPPED],
        "tailored_today": tailored_today,
        "ready": counts[JobStatus.READY],
        "submitted": counts[JobStatus.SUBMITTED],
        "failed": counts[JobStatus.FAILED],
        "expired": counts[JobStatus.EXPIRED],
        "skipped": counts[JobStatus.SKIPPED],
        "last_scan_time": meta.get("last_scan_time"),
        "groq_api_usage": meta.get("groq_api_usage", 0)
    }
