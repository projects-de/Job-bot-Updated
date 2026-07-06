import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from apify_client import ApifyClient
from src import config
from src.utils.logger import log, error

client = ApifyClient(config.APIFY_TOKEN)

def stable_job_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"

def normalize_dice_job(item: Dict) -> Dict:
    """
    Normalize a raw Dice job item from the "shahidirfan/Dice-Job-Scraper" Apify
    actor's dataset into the shared Job-Bot schema.
    """
    title = item.get("title") or ""
    location = item.get("location") or ""
    work_setting = (item.get("workSetting") or "").lower()

    is_remote = (
        "remote" in work_setting or
        bool(re.search(r"remote", title, re.IGNORECASE)) or
        bool(re.search(r"remote", location, re.IGNORECASE))
    )

    description = item.get("description_text") or item.get("summary") or ""
    description_html = item.get("description_html") or ""

    url = item.get("url") or item.get("detailsPageUrl") or ""
    raw_id = item.get("dice_id") or item.get("jobId") or item.get("id") or item.get("guid") or url or title

    return {
        "source": "Dice",
        "job_title": title or "Data Engineering",
        "company": item.get("company") or item.get("companyName") or "Unknown",
        "company_url": item.get("companyPageUrl"),
        "company_logo": item.get("companyLogoUrl"),
        "location": location or "United States",
        "remote": is_remote,
        "posted_date": item.get("posted") or item.get("updated") or item.get("firstActiveDate"),
        "scraped_at": datetime.utcnow().isoformat(),
        "url": url,
        "apply_url": url,
        "description": description,
        "description_html": description_html,
        "requirements": None,
        "salary": item.get("salary"),
        "employment_type": item.get("employmentType"),
        "id": stable_job_id("dice", str(raw_id)),
        "raw_payload": item,
    }

def fetch_dice_jobs_via_apify_api(options: Optional[Dict] = None) -> List[Dict]:
    """
    Fetch Dice.com jobs via the "shahidirfan/Dice-Job-Scraper" Apify actor.
    """
    if options is None:
        options = {}

    try:
        run_input = {
            "keyword": options.get("query", config.JOB_KEYWORDS),
            "location": options.get("location", config.JOB_LOCATION),
            "posted_date": "all",
            "results_wanted": options.get("results_wanted", 50),
            "proxyConfiguration": {"useApifyProxy": True},
        }

        log(f"Starting Dice Apify actor ({config.DICE_ACTOR_ID}) with input:")
        log(json.dumps(run_input, indent=2))

        run = client.actor(config.DICE_ACTOR_ID).call(
            run_input=run_input,
            run_timeout=timedelta(seconds=config.APIFY_TIMEOUT_SECONDS),
        )

        dataset_id = getattr(run, "default_dataset_id", None) or (run.get("defaultDatasetId") if isinstance(run, dict) else None)

        jobs = []
        if dataset_id:
            log(f"Actor run completed. Dataset ID: {dataset_id}")
            dataset_items = client.dataset(dataset_id).list_items().items

            for item in dataset_items:
                jobs.append(normalize_dice_job(item))
        else:
            error("Actor run completed but no dataset was returned.")

        log(f"Dice API fetch completed: {len(jobs)} jobs found")
        return jobs

    except Exception as err:
        error(f"Dice API fetch failed: {str(err)}")
        return []
