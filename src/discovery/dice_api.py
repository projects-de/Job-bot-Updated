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
    Normalize a raw Dice job item from the Apify dataset
    into the shared Job-Bot schema.
    """
    title = item.get("title") or ""
    location = item.get("location") or ""
    
    # Generic remote detection if not explicitly provided
    is_remote = (
        bool(re.search(r"remote", title, re.IGNORECASE)) or 
        bool(re.search(r"remote", location, re.IGNORECASE)) or
        item.get("isRemote", False)
    )

    return {
        "source": "Dice",
        "job_title": item.get("title", "Data Engineering"),
        "company": item.get("companyName", item.get("company", "Unknown")),
        "company_url": item.get("companyUrl"),
        "company_logo": item.get("companyLogo"),
        "location": location or "United States",
        "remote": is_remote,
        "posted_date": item.get("postedDate") or item.get("date"),
        "scraped_at": item.get("scrapedAt", datetime.utcnow().isoformat()),
        "url": item.get("url"),
        "apply_url": item.get("applyUrl") or item.get("url"),
        "description": item.get("description", ""),
        "description_html": item.get("descriptionHtml", ""),
        "requirements": None,
        "salary": item.get("salary"),
        "employment_type": item.get("employmentType"),
        "id": stable_job_id("dice", str(item.get("id") or item.get("url") or item.get("title") or "")),
        "raw_payload": item,
    }

def fetch_dice_jobs_via_apify_api(options: Optional[Dict] = None) -> List[Dict]:
    """
    Fetch Dice.com jobs via the Apify API.
    Uses the configured DICE_ACTOR_ID.
    """
    if options is None:
        options = {}

    try:
        from urllib.parse import quote
        q = quote(options.get("query", config.JOB_KEYWORDS))
        l = quote(options.get("location", config.JOB_LOCATION))
        search_url = f"https://www.dice.com/jobs?q={q}&location={l}"

        run_input = {
            "startUrls": [{"url": search_url}],
            "maxItems": options.get("maxItems", 50),
            "proxy": {"useApifyProxy": True}
        }

        log(f"Starting Dice Apify actor ({config.DICE_ACTOR_ID}) with input:")
        log(json.dumps(run_input, indent=2))

        run = client.actor(config.DICE_ACTOR_ID).call(
            run_input=run_input,
            run_timeout=timedelta(seconds=config.APIFY_TIMEOUT_SECONDS),
        )

        jobs = []
        if run and run.get("defaultDatasetId"):
            log(f"Actor run completed. Dataset ID: {run['defaultDatasetId']}")
            dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
            
            for item in dataset_items:
                jobs.append(normalize_dice_job(item))
        else:
            error("Actor run completed but no dataset was returned.")

        log(f"Dice API fetch completed: {len(jobs)} jobs found")
        return jobs

    except Exception as err:
        error(f"Dice API fetch failed: {str(err)}")
        return []
