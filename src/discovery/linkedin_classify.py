"""
Discovery-time classifier: visits each scraped LinkedIn job page once and tags
it 'easy_apply' or 'external' based on which Apply button is actually present,
using the exact same selectors the live apply flow uses
(src/workflow/linkedin_apply.py's EASY_APPLY_SELECTORS / EXTERNAL_APPLY_SELECTORS).

On the same page visit, this also reads LinkedIn's own workplace-type chip
(the "Hybrid"/"Remote"/"On-site" pill shown right on the job page, separate
from the description) — see WORKPLACE_CHIP_SELECTORS below — and trusts it
over the text-based classifier (src/utils/workplace_classifier) when both are
available, since it's LinkedIn's own stated value rather than an inference
from free text.

For an 'external' job where NEITHER the chip nor LinkedIn's own description
gave a signal, this also follows the real external Apply link — confirmed
live that the actual "Remote" statement sometimes only exists on the
employer's own careers/ATS page — and runs the same text classifier against
that page too, as a lower-confidence, gap-filling last resort.

Runs after discovery, in the background. Results are cached by stable job id
in data/linkedin_apply_type_cache.json so a job already classified is never
re-visited on a later scan, and written back into the jobs-all/jobs-unique
files the frontend already polls.
"""
import json
import os
import time
from glob import glob
from urllib.parse import urlparse, parse_qs

from selenium import webdriver
from selenium.common.exceptions import SessionNotCreatedException

from src.workflow.linkedin_apply import _find, EASY_APPLY_SELECTORS, EXTERNAL_APPLY_SELECTORS
from src.utils.workplace_classifier import classify_workplace_type_from_html, workplace_type_of

DATA_DIR = "data"
CACHE_PATH = os.path.join(DATA_DIR, "linkedin_apply_type_cache.json")
VISIT_PACING_SECONDS = 2.0
# External careers/ATS pages aren't a warmed-up, already-logged-in session
# like LinkedIn's own — many (Workday, Greenhouse, Lever) render their real
# content via JS after load, so this is deliberately longer than the 2.5s
# used for LinkedIn's own page.
EXTERNAL_PAGE_RENDER_WAIT_SECONDS = 3.5

# LinkedIn's own job page shows the workplace type as a small standalone
# pill/chip right in the top card (e.g. "Hybrid" next to "Full-time"),
# separate from the description text entirely — confirmed live via
# screenshot. This is a direct, structured read of that same chip, and is
# trusted OVER the text-based classifier (src/utils/workplace_classifier)
# when both are available, since it's LinkedIn's own stated value rather
# than an inference from free text. XPath text-matching (not a guessed CSS
# class) is used deliberately: LinkedIn's class names churn often, but the
# chip's own text content ("Hybrid"/"Remote"/"On-site") is the one stable
# thing about it. normalize-space(text()) requires the element's own text to
# be EXACTLY that word — not just contain it — so a description paragraph
# that happens to mention "Hybrid" can never match this.
WORKPLACE_CHIP_SELECTORS = [
    f"xpath://{tag}[normalize-space(text())='{value}']"
    for value in ("Hybrid", "Remote", "On-site", "Onsite", "On-Site")
    for tag in ("span", "button", "li", "div")
]


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


def _resolve_external_url(el) -> str:
    """
    Get the real employer destination from the matched External Apply
    element. Per the selector comments in linkedin_apply.py, LinkedIn's
    current layout wraps the real link in a linkedin.com/safety/go/?url=...
    redirect — unwrap that to the real URL rather than following the
    redirect with a second page load. Returns "" if the element has no href
    (a <button>-based match, or the wrapper couldn't be parsed) — that's a
    normal miss, not an error, and just means no external check is possible.
    """
    try:
        href = el.get_attribute("href") or ""
    except Exception:
        return ""
    if not href:
        return ""
    if "linkedin.com/safety/go" in href:
        try:
            real = parse_qs(urlparse(href).query).get("url", [""])[0]
            return real or ""
        except Exception:
            return ""
    if "linkedin.com" in href:
        return ""  # Still a LinkedIn URL, not a real external destination
    return href


def _read_workplace_chip(driver) -> str:
    """
    Look for the workplace-type chip directly on the currently-loaded
    LinkedIn job page. Returns "" if not found — most job pages may render
    this differently or not at all, which is a normal miss, not an error;
    the text-based classifier remains the fallback.
    """
    try:
        el = _find(driver, WORKPLACE_CHIP_SELECTORS, timeout=3)
    except Exception:
        return ""
    if not el:
        return ""
    try:
        text = (el.text or "").strip().lower()
    except Exception:
        return ""
    if not text:
        return ""
    return workplace_type_of(text)


def _check_external_workplace_type(driver, external_url: str) -> str:
    """
    Visit the real employer page and run the same shared text classifier
    against it. Any failure (timeout, block, unreachable site, login wall)
    just returns "" — this is a best-effort enrichment on top of a job that
    already has no Remote/Hybrid/Onsite signal, never something that should
    interrupt the classification pass.
    """
    try:
        driver.get(external_url)
        time.sleep(EXTERNAL_PAGE_RENDER_WAIT_SECONDS)
        return classify_workplace_type_from_html(driver.page_source)
    except Exception as e:
        print(f"[LinkedIn Classify] External page check failed for {external_url}: {e}")
        return ""


def _classify_one(driver, url: str, existing_workplace_type: str = "") -> dict:
    driver.get(url)
    time.sleep(2.5)

    result = {}

    # LinkedIn's own chip is checked first, regardless of apply_type, and
    # marked "chip" so the merge step knows to trust it over whatever the
    # text classifier already found (see WORKPLACE_CHIP_SELECTORS above).
    chip_type = _read_workplace_chip(driver)
    if chip_type:
        result["workplace_type_enum"] = chip_type
        result["workplace_type_source"] = "chip"

    if _find(driver, EASY_APPLY_SELECTORS, timeout=4):
        result["apply_type"] = "easy_apply"
        return result

    external_el = _find(driver, EXTERNAL_APPLY_SELECTORS, timeout=4)
    if external_el:
        result["apply_type"] = "external"
        # Only fall through to the (slower, less reliable) external-page
        # check when neither the chip nor a prior scan already answered this.
        if not chip_type and not existing_workplace_type:
            external_url = _resolve_external_url(external_el)
            if external_url:
                workplace_type = _check_external_workplace_type(driver, external_url)
                if workplace_type:
                    result["workplace_type_enum"] = workplace_type
                    result["workplace_type_source"] = "external_text"
        return result

    result["apply_type"] = "unknown"
    return result


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
                pending_by_id[jid] = {"url": url, "workplace_type_enum": j.get("workplace_type_enum") or ""}

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
                for i, (jid, info) in enumerate(pending_by_id.items(), start=1):
                    try:
                        result = _classify_one(driver, info["url"], info["workplace_type_enum"])
                    except Exception as e:
                        print(f"[LinkedIn Classify] Error checking {info['url']}: {e}")
                        result = {"apply_type": "unknown"}
                    # Only cache when apply_type actually resolved — an "unknown"
                    # result (including one that still found a chip value) must
                    # stay OUT of the cache so this job's apply_type gets another
                    # attempt on the next scan instead of being stuck forever;
                    # a chip value found on this attempt just gets rediscovered
                    # then, at no real cost.
                    if result.get("apply_type") != "unknown":
                        cache[jid] = result
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
            if not jid or str(j.get("source", "")).lower() != "linkedin" or jid not in cache:
                continue
            # Cache entries were a plain apply_type string before the external
            # workplace-type check was added — normalize old entries so a
            # cache built before this feature doesn't need to be cleared.
            cached = cache[jid]
            if isinstance(cached, str):
                cached = {"apply_type": cached}
            if j.get("apply_type") != cached.get("apply_type"):
                j["apply_type"] = cached.get("apply_type")
                changed = True
            resolved_workplace_type = cached.get("workplace_type_enum")
            resolved_source = cached.get("workplace_type_source")
            if resolved_workplace_type:
                # A chip read is LinkedIn's own stated value — trust it over
                # whatever the text classifier guessed, even if that guess
                # already set something. An external-page text match stays
                # gap-filling only (never overrides an existing value), same
                # as before this feature existed.
                should_apply = (
                    resolved_source == "chip" and j.get("workplace_type_enum") != resolved_workplace_type
                ) or (resolved_source != "chip" and not j.get("workplace_type_enum"))
                if should_apply:
                    j["workplace_type_enum"] = resolved_workplace_type
                    j["remote"] = resolved_workplace_type == "REMOTE"
                    changed = True
        if changed:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(jobs, f, indent=2)
            except Exception as e:
                print(f"[LinkedIn Classify] Could not write {path}: {e}")
