import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from apify_client import ApifyClient

from src import config
from src.utils.logger import error, log

client = ApifyClient(config.APIFY_TOKEN)

ZIPRECRUITER_ACTOR_ID = "apify/ziprecruiter-scraper"


def stable_job_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _first_text(item: Dict, keys: List[str], default: str = "") -> str:
    for key in keys:
        value = item.get(key)
        if value:
            return str(value).strip()
    return default


def normalize_ziprecruiter_job(item: Dict) -> Dict:
    title = _first_text(item, ["jobTitle", "title", "positionName"], "Data Engineering")
    company = _first_text(item, ["company", "companyName", "employerName"], "Unknown")
    location = _first_text(item, ["location", "jobLocation"], config.JOB_LOCATION)
    description = _first_text(item, ["description", "snippet", "summary", "jobDescription"])
    posted_date = _first_text(item, ["datePosted", "postedDate", "postedAt", "age"], "")
    url = _first_text(item, ["jobUrl", "url", "link"], "")
    salary = _first_text(item, ["salary", "salaryRange"], "") or None

    remote = bool(item.get("remote")) or "remote" in location.lower()

    return {
        "source": "ZipRecruiter",
        "job_title": title,
        "company": company,
        "location": location,
        "remote": remote,
        "posted_date": posted_date or datetime.utcnow().isoformat(),
        "scraped_at": item.get("scrapedAt", datetime.utcnow().isoformat()),
        "url": url,
        "apply_url": url,
        "description": description,
        "description_html": _first_text(item, ["descriptionHtml", "html"], ""),
        "requirements": None,
        "salary": salary,
        "id": stable_job_id("ziprecruiter", str(item.get("id") or url or title)),
        "raw_payload": item,
    }


def fetch_ziprecruiter_jobs_via_apify_api(options: Optional[Dict] = None) -> List[Dict]:
    if options is None:
        options = {}

    try:
        run_input = {
            "search": options.get("search", config.JOB_KEYWORDS),
            "location": options.get("location", config.JOB_LOCATION),
            "radius": options.get("radius", 0),
            "days": options.get("days", config.MAX_AGE_DAYS),
            "remote": options.get("remote", True),
            "maxPages": options.get("maxPages", 1),
        }

        log(f"Starting ZipRecruiter Apify actor ({ZIPRECRUITER_ACTOR_ID}) with input:")
        log(json.dumps(run_input, indent=2))

        run = client.actor(ZIPRECRUITER_ACTOR_ID).call(
            run_input=run_input,
            run_timeout=timedelta(seconds=config.APIFY_TIMEOUT_SECONDS),
            memory_mbytes=256,
        )

        jobs = []
        if run and run.get("defaultDatasetId"):
            log(f"ZipRecruiter actor completed. Dataset ID: {run['defaultDatasetId']}")
            dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
            for item in dataset_items:
                jobs.append(normalize_ziprecruiter_job(item))
        else:
            error("ZipRecruiter actor completed but no dataset was returned.")

        log(f"ZipRecruiter API fetch completed: {len(jobs)} jobs found")
        return jobs
    except Exception as err:
        error(f"ZipRecruiter API fetch failed: {str(err)}")
        return []
