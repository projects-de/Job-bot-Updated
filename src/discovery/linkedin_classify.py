"""
Discovery-time classifier: visits each scraped LinkedIn job page once and tags
it 'easy_apply' or 'external' based on which Apply button is actually present,
using the exact same selectors the live apply flow uses
(src/workflow/linkedin_apply.py's EASY_APPLY_SELECTORS / EXTERNAL_APPLY_SELECTORS).

Runs after discovery, in the background. Results are cached by stable job id
in data/linkedin_apply_type_cache.json so a job already classified is never
re-visited on a later scan, and written back into the jobs-all/jobs-unique
files the frontend already polls.
"""
import json
import os
import time
from glob import glob

from selenium import webdriver
from selenium.common.exceptions import SessionNotCreatedException

from src.workflow.linkedin_apply import _find, EASY_APPLY_SELECTORS, EXTERNAL_APPLY_SELECTORS

DATA_DIR = "data"
CACHE_PATH = os.path.join(DATA_DIR, "linkedin_apply_type_cache.json")
VISIT_PACING_SECONDS = 2.0


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _build_driver():
    # Same profile/options as SequentialApplier.init_browser (src/workflow/apply.py)
    # so LinkedIn sees an already-logged-in session — required for the Apply
    # button to render at all instead of a login wall.
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--no-first-run')
    options.add_argument('--no-default-browser-check')

    user_data_dir = os.path.join(os.getcwd(), "user_data_uc")
    os.makedirs(user_data_dir, exist_ok=True)
    options.add_argument(f"--user-data-dir={user_data_dir}")

    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
    })
    return driver


def _classify_one(driver, url: str) -> str:
    driver.get(url)
    time.sleep(2.5)
    if _find(driver, EASY_APPLY_SELECTORS, timeout=4):
        return "easy_apply"
    if _find(driver, EXTERNAL_APPLY_SELECTORS, timeout=4):
        return "external"
    return "unknown"


def classify_linkedin_jobs(progress_callback=None) -> None:
    """
    Visit every not-yet-classified LinkedIn job in the latest jobs-all/
    jobs-unique files and tag it with apply_type. Safe to call after every
    scan: already-classified jobs (by id, in the on-disk cache) are skipped
    entirely, so repeat scans only pay the cost of genuinely new postings.
    """
    cache = _load_cache()

    jobs_by_file = {}
    for suffix in ("all", "unique"):
        files = sorted(glob(os.path.join(DATA_DIR, f"jobs-{suffix}-*.json")), reverse=True)
        if not files:
            continue
        path = files[0]
        try:
            with open(path, "r", encoding="utf-8") as f:
                jobs_by_file[path] = json.load(f)
        except Exception as e:
            print(f"[LinkedIn Classify] Could not read {path}: {e}")

    pending_by_id = {}
    for jobs in jobs_by_file.values():
        for j in jobs:
            jid = str(j.get("id") or "").strip()
            url = j.get("url")
            if jid and url and str(j.get("source", "")).lower() == "linkedin" and jid not in cache:
                pending_by_id[jid] = url

    if pending_by_id:
        driver = None
        try:
            driver = _build_driver()
        except SessionNotCreatedException as e:
            if "already in use" in str(e).lower():
                print("[LinkedIn Classify] Apply session is currently using this Chrome profile — "
                      "skipping Easy Apply classification this run (will retry on the next scan).")
            else:
                print(f"[LinkedIn Classify] Could not start Chrome: {e}")
            driver = None

        if driver:
            try:
                total = len(pending_by_id)
                for i, (jid, url) in enumerate(pending_by_id.items(), start=1):
                    try:
                        apply_type = _classify_one(driver, url)
                    except Exception as e:
                        print(f"[LinkedIn Classify] Error checking {url}: {e}")
                        apply_type = "unknown"
                    if apply_type != "unknown":
                        cache[jid] = apply_type
                        _save_cache(cache)
                    if progress_callback:
                        try:
                            progress_callback(i, total)
                        except Exception:
                            pass
                    time.sleep(VISIT_PACING_SECONDS)
            finally:
                try:
                    driver.quit()
                except Exception:
                    pass

    for path, jobs in jobs_by_file.items():
        changed = False
        for j in jobs:
            jid = str(j.get("id") or "").strip()
            if jid and str(j.get("source", "")).lower() == "linkedin" and jid in cache:
                if j.get("apply_type") != cache[jid]:
                    j["apply_type"] = cache[jid]
                    changed = True
        if changed:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(jobs, f, indent=2)
            except Exception as e:
                print(f"[LinkedIn Classify] Could not write {path}: {e}")
