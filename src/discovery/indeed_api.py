import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from apify_client import ApifyClient
from src import config
from src.utils.logger import log, error

client = ApifyClient(config.APIFY_TOKEN)

# Actor ID for the Indeed Jobs Scraper by valig
INDEED_ACTOR_ID = "valig/indeed-jobs-scraper"


def stable_job_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def normalize_indeed_job(item: Dict) -> Dict:
    """
    Normalize a raw Indeed job item from the Apify dataset
    into the shared Job-Bot schema.
    """
    # Location normalization
    loc_data = item.get("location", {})
    city = loc_data.get("city", "")
    state = loc_data.get("admin1Code", "")
    location_str = f"{city}, {state}" if city and state else (city or state or "United States")
    
    # Salary normalization
    salary_data = item.get("baseSalary", {})
    salary_str = None
    if salary_data.get("min") or salary_data.get("max"):
        min_v = salary_data.get("min")
        max_v = salary_data.get("max")
        currency = salary_data.get("currencyCode", "$")
        if min_v and max_v:
            salary_str = f"{currency}{min_v} - {currency}{max_v}"
        elif min_v:
            salary_str = f"{currency}{min_v}+"
        elif max_v:
            salary_str = f"Up to {currency}{max_v}"

    # Company
    employer = item.get("employer", {})
    company_name = employer.get("name", "Unknown")

    return {
        "source": "Indeed",
        "job_title": item.get("title", "Data Engineering"),
        "company": company_name,
        "company_url": employer.get("companyPageUrl"),
        "location": location_str,
        "remote": config.REMOTE_ONLY, # Indeed filters are usually precise
        "posted_date": item.get("datePublished"),
        "scraped_at": item.get("dateOnIndeed", datetime.utcnow().isoformat()),
        "url": item.get("url"),
        "apply_url": item.get("jobUrl") or item.get("url"),
        "description": item.get("description", {}).get("text", ""),
        "description_html": item.get("description", {}).get("text", ""), # Indeed scraper often provides only text or mixed
        "requirements": None,
        "salary": salary_str,
        "id": stable_job_id("indeed", str(item.get("key") or item.get("url") or item.get("title") or "")),
        "raw_payload": item,
    }

def fetch_indeed_jobs_via_apify_api(options: Optional[Dict] = None) -> List[Dict]:
    """
    Fetch Indeed Data Engineering jobs via the Apify API.
    Uses the valig/indeed-jobs-scraper actor.

    Indeed is far more aggressive than LinkedIn/Dice about blocking automated
    scraping, so the actor can legitimately complete successfully (no
    exception, no error status) but still return an empty dataset for a
    single attempt if that run's request got blocked/rate-limited on Indeed's
    side. A retry a few seconds later is usually enough to recover, so we
    retry once on an empty (but non-erroring) result before giving up.
    """
    if options is None:
        options = {}

    run_input = {
        "title": options.get("title", config.JOB_KEYWORDS),
        "location": options.get("location", config.JOB_LOCATION),
        "country": options.get("country", "us"),
        "datePosted": options.get("datePosted", "1"), # Last 24 hours (Indeed internal value)
        "limit": options.get("limit", 50),
    }

    log(f"Starting Indeed Apify actor ({INDEED_ACTOR_ID}) with input:")
    log(json.dumps(run_input, indent=2))

    max_attempts = 2
    retry_wait_seconds = 10
    last_problem = None

    for attempt in range(1, max_attempts + 1):
        try:
            run = client.actor(INDEED_ACTOR_ID).call(
                run_input=run_input,
                run_timeout=timedelta(seconds=config.APIFY_TIMEOUT_SECONDS),
            )

            dataset_id = getattr(run, "default_dataset_id", None) or (run.get("defaultDatasetId") if isinstance(run, dict) else None)
            if not dataset_id:
                last_problem = "actor run completed but no dataset was returned"
                error(f"Indeed API fetch attempt {attempt}/{max_attempts}: {last_problem}")
            else:
                log(f"Indeed Actor run completed (attempt {attempt}/{max_attempts}). Dataset ID: {dataset_id}")
                dataset_items = client.dataset(dataset_id).list_items().items

                if dataset_items:
                    jobs = [normalize_indeed_job(item) for item in dataset_items]
                    log(f"Indeed API fetch completed: {len(jobs)} jobs found")
                    return jobs

                last_problem = "actor run succeeded but returned 0 jobs (likely blocked/rate-limited by Indeed for this attempt)"
                log(f"Indeed API fetch attempt {attempt}/{max_attempts}: {last_problem}")
        except Exception as err:
            last_problem = str(err)
            error(f"Indeed API fetch attempt {attempt}/{max_attempts} failed: {last_problem}")

        if attempt < max_attempts:
            log(f"Retrying Indeed fetch in {retry_wait_seconds}s...")
            time.sleep(retry_wait_seconds)

    error(f"Indeed API fetch failed after {max_attempts} attempts: {last_problem}")
    return []
