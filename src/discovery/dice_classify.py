"""
Discovery-time classifier: visits each scraped Dice job page once and tags it
'easy_apply' or 'external' based on the destination of Dice's own "Apply Now"
link (data-testid="apply-button"), whose href reveals the flow without ever
needing to click it or be logged in:
  - .../job-applications/{id}/wizard      -> Dice's own native apply wizard
  - .../job-applications/{id}/start-apply -> hands off to an external site
    (matches the "Checking for external redirect..." handling already in
    src/workflow/dice_apply.py)
Confirmed live across 30 real Dice job postings.

Runs after discovery, in the background. Results are cached by stable job id
in data/dice_apply_type_cache.json so a job already classified is never
re-visited on a later scan, and written back into the jobs-all/jobs-unique
files the frontend already polls. Mirrors src/discovery/linkedin_classify.py.
"""
import json
import os
import time
from glob import glob

from selenium import webdriver
from selenium.common.exceptions import SessionNotCreatedException

DATA_DIR = "data"
CACHE_PATH = os.path.join(DATA_DIR, "dice_apply_type_cache.json")
VISIT_PACING_SECONDS = 1.5

APPLY_BUTTON_HREF_JS = """
    const el = document.querySelector('[data-testid="apply-button"]');
    return el ? el.getAttribute('href') : null;
"""


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
    # No persistent profile needed here — unlike LinkedIn, a Dice job page's
    # Apply button href is visible in the public page without being logged
    # in, confirmed live. Using a fresh/default profile also means this can
    # never collide with the real apply flow's own Chrome profile lock.
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--no-first-run')
    options.add_argument('--no-default-browser-check')

    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
    })
    return driver


def _classify_one(driver, url: str) -> str:
    driver.get(url)
    time.sleep(3)
    href = driver.execute_script(APPLY_BUTTON_HREF_JS)
    if not href:
        print(f"[Dice Classify] No apply-button href found for {url} (page title: {driver.title!r})")
        return "unknown"
    # The href is URL-encoded (e.g. "...%2Fjob-applications%2F{id}%2Fwizard"),
    # so there's never a literal "/wizard" substring — only "wizard" itself,
    # confirmed live: an earlier version of this check used "/wizard" and
    # "/start-apply" and silently misclassified every single job as unknown.
    if "wizard" in href:
        return "easy_apply"
    if "start-apply" in href:
        return "external"
    print(f"[Dice Classify] Unrecognized apply-button href pattern for {url}: {href!r}")
    return "unknown"


def classify_dice_jobs(progress_callback=None) -> None:
    """
    Visit every not-yet-classified Dice job in the latest jobs-all/
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
            print(f"[Dice Classify] Could not read {path}: {e}")

    pending_by_id = {}
    for jobs in jobs_by_file.values():
        for j in jobs:
            jid = str(j.get("id") or "").strip()
            url = j.get("url")
            if jid and url and str(j.get("source", "")).lower() == "dice" and jid not in cache:
                pending_by_id[jid] = url

    if pending_by_id:
        driver = None
        try:
            driver = _build_driver()
        except SessionNotCreatedException as e:
            if "already in use" in str(e).lower():
                print("[Dice Classify] Apply session is currently using this Chrome profile — "
                      "skipping Dice Easy Apply classification this run (will retry on the next scan).")
            else:
                print(f"[Dice Classify] Could not start Chrome: {e}")
            driver = None

        if driver:
            try:
                total = len(pending_by_id)
                for i, (jid, url) in enumerate(pending_by_id.items(), start=1):
                    try:
                        apply_type = _classify_one(driver, url)
                    except Exception as e:
                        print(f"[Dice Classify] Error checking {url}: {e}")
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
            if jid and str(j.get("source", "")).lower() == "dice" and jid in cache:
                if j.get("apply_type") != cache[jid]:
                    j["apply_type"] = cache[jid]
                    changed = True
        if changed:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(jobs, f, indent=2)
            except Exception as e:
                print(f"[Dice Classify] Could not write {path}: {e}")
