import sys
import os
import json
from glob import glob
from contextlib import redirect_stdout, redirect_stderr
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from datetime import datetime
import shutil

# Add the project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Force UTF-8 encoding for Windows terminals to prevent 'charmap' crashes
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # Fix for asyncio 'WinError 10054' crash on Windows with ProactorEventLoop
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass

from src.workflow.discover import run_discovery
from src.utils.cv_processor import process_cv
from src.workflow.optimize import generate_cover_letter, render_tailored_cv_latex
from src.workflow.apply import signal_review_complete
from src.storage.tracker import (
    get_tracker_summary, 
    save_parsed_profile, 
    save_tailored_application,
    get_dashboard_stats,
    JobStatus,
    UPLOADS_DIR, 
    PROFILES_DIR,
    APPLICATIONS_DIR,
    DATA_DIR
)

app = FastAPI(title="Job-Bot API")

# Simple state to track background processes
discovery_status = {"running": False, "last_run": None, "error": None, "last_message": None}
application_status = {"running": False, "current_job": None, "error": None}
DISCOVERY_LOG_FILE = os.path.join(BASE_DIR, "scratch", "discover.log")


class _TeeWriter:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
    def flush(self):
        for stream in self.streams:
            if hasattr(stream, 'flush'):
                stream.flush()

class _LogCallbackStream:
    def __init__(self, callback):
        self.callback = callback
    def write(self, data):
        if data.strip():
            self.callback(data)
    def flush(self):
        pass

def _append_discovery_log(message: str):
    os.makedirs(os.path.dirname(DISCOVERY_LOG_FILE), exist_ok=True)
    # Open with 'a' and close immediately to avoid locking
    try:
        with open(DISCOVERY_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message.rstrip("\n") + "\n")
    except Exception:
        pass # Ignore transient lock errors during write


def _reset_discovery_log():
    os.makedirs(os.path.dirname(DISCOVERY_LOG_FILE), exist_ok=True)
    with open(DISCOVERY_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

frontend_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if origin.strip()
]

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_latest_jobs_file(mode="unique"):
    suffix = "-all" if mode == "all" else "-unique"
    files = glob(os.path.join(DATA_DIR, f"jobs{suffix}-*.json"))
    
    if not files:
        files = glob(os.path.join(DATA_DIR, "jobs-*.json"))
        
    if not files:
        return None
    files.sort(reverse=True)
    return files[0]


def _load_jobs_from_file(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_job_index() -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for mode in ("all", "unique"):
        latest_file = get_latest_jobs_file(mode)
        for job in _load_jobs_from_file(latest_file):
            jid = str(job.get("id") or job.get("job_id") or "").strip()
            if jid and jid not in indexed:
                indexed[jid] = job
    return indexed


def _load_tailored_payload(job_id: str) -> Dict[str, Any]:
    filename = os.path.join(APPLICATIONS_DIR, job_id, "tailored_cv.json")
    if not os.path.exists(filename):
        return {}
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/jobs")
async def get_jobs(mode: str = "unique"):
    latest_file = get_latest_jobs_file(mode)
    if not latest_file:
        return []
    
    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            jobs = json.load(f)
            
        # Merge with tracker status
        tracker = get_tracker_summary()
        status_map = {a["job_id"]: a["status"] for a in tracker.get("applications", [])}
        
        for job in jobs:
            jid = str(job.get('id') or job.get('job_id'))
            job["status"] = status_map.get(jid, JobStatus.DISCOVERED)
            
        return jobs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/applied")
async def get_applied_jobs():
    try:
        tracker = get_tracker_summary()
        applications = tracker.get("applications", [])
        job_index = _build_job_index()

        applied_jobs = []
        terminal_statuses = (JobStatus.SUBMITTED, JobStatus.EXPIRED, JobStatus.SKIPPED)
        for app_entry in applications:
            if app_entry.get("status") not in terminal_statuses:
                continue

            job_id = str(app_entry.get("job_id") or "").strip()
            if not job_id:
                continue

            job_details = job_index.get(job_id, {})
            tailored = _load_tailored_payload(job_id)
            applied_jobs.append(
                {
                    "job_id": job_id,
                    "job_title": app_entry.get("job_title") or job_details.get("job_title", ""),
                    "company": app_entry.get("company") or job_details.get("company", ""),
                    "status": app_entry.get("status", JobStatus.SUBMITTED),
                    "submitted_at": app_entry.get("updated_at"),
                    "job_url": job_details.get("url") or job_details.get("job_url", ""),
                    "source": job_details.get("source", ""),
                    "location": job_details.get("location", ""),
                    "posted_date": job_details.get("posted_date", ""),
                    "tailored_cv": tailored.get("cv"),
                    "tailored_cv_latex": tailored.get("cv_latex", ""),
                    "cover_letter": tailored.get("cover_letter"),
                }
            )

        applied_jobs.sort(key=lambda x: x.get("submitted_at") or "", reverse=True)
        return applied_jobs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    return get_dashboard_stats()

@app.get("/summary")
async def get_summary():
    # Get latest static summary
    files = glob(os.path.join(DATA_DIR, "summary-*.json"))
    summary_data = {}
    if files:
        files.sort(reverse=True)
        try:
            with open(files[0], "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except:
            pass
            
    # Get dynamic stats from tracker
    from src.storage.tracker import get_dashboard_stats
    tracker_stats = get_dashboard_stats()
    
    # Merge
    return {**summary_data, **tracker_stats}


@app.get("/health")
async def get_health():
    return {
        "ok": True,
        "discovery_running": discovery_status["running"],
        "application_running": application_status["running"],
    }

def run_discovery_task():
    discovery_status["running"] = True
    discovery_status["error"] = None
    discovery_status["last_message"] = "Starting discovery"
    _reset_discovery_log()
    try:
        # Use a callback stream to avoid keeping the file open
        log_stream = _LogCallbackStream(_append_discovery_log)
        tee = _TeeWriter(sys.stdout, log_stream)
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f"[{datetime.now().isoformat()}] Discovery task started")
            run_discovery()
        discovery_status["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        discovery_status["last_message"] = "Discovery complete"
        from src.storage.tracker import save_last_scan_time
        save_last_scan_time()
    except Exception as e:
        discovery_status["error"] = str(e)
        discovery_status["last_message"] = f"Discovery failed: {e}"
        _append_discovery_log(f"[ERROR {datetime.utcnow().isoformat()}] {e}")
    finally:
        discovery_status["running"] = False

@app.post("/discover")
async def start_discovery(background_tasks: BackgroundTasks):
    # Check if already running to avoid double-starting
    if discovery_status.get("running"):
        return {"message": "Discovery is already in progress"}

    # Set status to TRUE immediately before starting background task
    discovery_status["running"] = True
    discovery_status["last_message"] = "Starting discovery pipeline..."
    
    background_tasks.add_task(run_discovery_task)
    return {"message": "Discovery started in background"}

@app.get("/discover/status")
async def get_discovery_status():
    return discovery_status


@app.get("/discover/logs")
async def get_discovery_logs(limit: int = 25):
    if not os.path.exists(DISCOVERY_LOG_FILE):
        return {"lines": []}

    try:
        with open(DISCOVERY_LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except Exception:
        # Ignore transient read locks or errors
        return {"lines": []}

    return {"lines": lines[-max(1, min(limit, 200)) :]}

@app.post("/cv/upload")
async def upload_cv(file: UploadFile = File(...)):
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    file_path = os.path.join(UPLOADS_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        profile = process_cv(file_path)
        if not profile:
            raise ValueError("LLM failed to extract structured data from CV")
            
        save_parsed_profile(profile)
        return {"message": "CV processed successfully", "profile": profile}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cv/clear")
async def clear_cv():
    # Remove parsed profile
    profile_path = os.path.join(PROFILES_DIR, "parsed_profile.json")
    if os.path.exists(profile_path):
        try:
            os.remove(profile_path)
        except Exception:
            pass
            
    # Clear uploads directory
    try:
        for f in glob(os.path.join(UPLOADS_DIR, "*")):
            os.remove(f)
    except Exception:
        pass
        
    return {"message": "CV cleared"}

@app.get("/cv/file")
async def get_cv_file():
    files = glob(os.path.join(UPLOADS_DIR, "*"))
    if not files:
        raise HTTPException(status_code=404, detail="No CV uploaded")
    files.sort(key=os.path.getmtime, reverse=True)
    latest = files[0]
    ext = os.path.splitext(latest)[1].lower()
    media_type_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
    }
    from fastapi.responses import FileResponse
    return FileResponse(
        latest,
        media_type=media_type_map.get(ext, "application/octet-stream"),
        content_disposition_type="inline",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@app.get("/cv/profile")
def get_cv_profile():
    profile_path = os.path.join(PROFILES_DIR, "parsed_profile.json")
    if not os.path.exists(profile_path):
        return {}
    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/jobs/tailored/{job_id}")
async def get_tailored_details(job_id: str):
    tailored_payload = _load_tailored_payload(job_id)
    if not tailored_payload:
        raise HTTPException(status_code=404, detail="No tailoring found for this job")
    return tailored_payload

@app.post("/jobs/tailor/{job_id}")
def tailor_job(job_id: str):
    latest_unique_file = get_latest_jobs_file("unique")
    latest_all_file = get_latest_jobs_file("all")
    if not latest_unique_file and not latest_all_file:
        raise HTTPException(status_code=404, detail="No jobs discovered yet")

    jobs = _load_jobs_from_file(latest_unique_file) + _load_jobs_from_file(latest_all_file)
    job = next((j for j in jobs if str(j.get('id') or j.get('job_id')) == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    # Get profile
    profile = get_cv_profile()
    if not profile:
        raise HTTPException(status_code=400, detail="CV not processed yet")
        
    try:
        # Truncate description to avoid token limits on the Groq call
        description = (job.get('description') or '')[:4000]

        # CV is used as-is (no LLM rewriting) — only the cover letter is generated
        # per job description, per explicit instruction to stop tailoring the CV.
        t_cv = profile

        print(f"[PROCESS] Generating cover letter for job: {job_id}...")
        t_cl = generate_cover_letter(profile, description)

        print(f"[PROCESS] Rendering LaTeX version...")
        t_latex = render_tailored_cv_latex(t_cv)
        
        payload = {"cv": t_cv, "cv_latex": t_latex, "cover_letter": t_cl}
        save_tailored_application(job_id, payload)
        
        # Log to tracker
        from src.storage.tracker import log_application, JobStatus
        log_application(job_id, job.get('job_title', 'Unknown'), job.get('company', 'Unknown'), JobStatus.TAILORED)
        
        print(f"[OK] Tailoring complete for: {job_id}")
        return {"message": "Job tailored successfully", "job_id": job_id, "tailored": payload}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/apply/status")
async def get_apply_status():
    import time as _time
    from src.workflow.apply import review_deadline_epoch, REVIEW_TIMEOUT_SECONDS

    deadline = review_deadline_epoch["value"]
    seconds_remaining = max(0, round(deadline - _time.time())) if deadline else None

    return {
        **application_status,
        "review_timeout_seconds": REVIEW_TIMEOUT_SECONDS,
        "seconds_remaining": seconds_remaining,
    }

@app.post("/apply/review/complete")
async def review_complete():
    signal_review_complete()
    return {"message": "Review signal sent"}

@app.post("/apply/skip")
async def apply_skip():
    from src.workflow.apply import signal_skip
    signal_skip()
    return {"message": "Skip signal sent"}

@app.get("/apply/tracker")
async def get_apply_tracker():
    return get_tracker_summary()


def run_apply_task(selected_jobs, profile):
    application_status["running"] = True
    application_status["total"] = len(selected_jobs)
    application_status["current_idx"] = 0
    application_status["error"] = None
    
    try:
        from src.workflow.apply import SequentialApplier
        from src.workflow.optimize import generate_cover_letter

        applier = SequentialApplier(profile, generate_cover_letter)
        try:
            for i, job in enumerate(selected_jobs):
                application_status["current_idx"] = i + 1
                application_status["current_job"] = {
                    "id": str(job.get('id') or job.get('job_id') or ""),
                    "title": job.get('job_title'),
                    "company": job.get('company')
                }
                applier.apply_to_job(job)
        finally:
            applier.close()
    except Exception as e:
        application_status["error"] = str(e)
    finally:
        application_status["running"] = False
        application_status["current_job"] = None

async def start_applications_impl(job_ids: List[str], background_tasks: BackgroundTasks):
    print(f"DEBUG: start_applications_impl called with job_ids: {job_ids}")
    if application_status["running"]:
        print("DEBUG: applications already in progress")
        return {"message": "Applications are already in progress"}

    latest_unique_file = get_latest_jobs_file("unique")
    latest_all_file = get_latest_jobs_file("all")
    print(f"DEBUG: latest unique: {latest_unique_file}, latest all: {latest_all_file}")
    if not latest_unique_file and not latest_all_file:
        raise HTTPException(status_code=404, detail="No jobs found")
        
    jobs = _load_jobs_from_file(latest_unique_file) + _load_jobs_from_file(latest_all_file)
    job_map = {}
    for j in jobs:
        jid = str(j.get('id') or j.get('job_id') or "")
        if jid and jid not in job_map:
            job_map[jid] = j

    selected_jobs = [job_map[jid] for jid in job_ids if jid in job_map]
    print(f"DEBUG: matched selected_jobs count: {len(selected_jobs)}")

    if not selected_jobs:
        print("DEBUG: raising 404 No valid jobs selected")
        raise HTTPException(status_code=404, detail="No valid jobs selected")

    profile = get_cv_profile()
    print(f"DEBUG: profile loaded: {bool(profile)}")

    # Profile is optional if tailored data already exists for the job.
    # The apply workflow will load pre-generated tailored docs from disk.
    # We still need at least an empty dict so the applier can fall back gracefully.
    if not profile:
        # Check if all selected jobs have pre-generated tailored data
        all_have_tailored = all(
            os.path.exists(os.path.join(APPLICATIONS_DIR, str(j.get('id') or j.get('job_id')), "tailored_cv.json"))
            for j in selected_jobs
        )
        if all_have_tailored:
            print("DEBUG: No profile but tailored data exists, proceeding with empty profile")
            profile = {}
        else:
            print("DEBUG: raising 400 CV not processed yet")
            raise HTTPException(status_code=400, detail="CV not processed yet. Please upload your CV first.")

    background_tasks.add_task(run_apply_task, selected_jobs, profile)
    print("DEBUG: background task added successfully")
    return {"message": f"Sequential application for {len(selected_jobs)} jobs started."}


@app.post("/apply/start")
@app.post("/apply/bulk")
async def start_applications(job_ids: List[str], background_tasks: BackgroundTasks):
    try:
        return await start_applications_impl(job_ids, background_tasks)
    except Exception as e:
        print(f"DEBUG Exception in start_applications: {e}")
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
