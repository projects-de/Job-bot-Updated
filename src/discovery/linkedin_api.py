import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from apify_client import ApifyClient
from src import config
from src.utils.logger import log, error

client = ApifyClient(config.APIFY_TOKEN)

# Actor ID for the LinkedIn Jobs Scraper by automation-lab
LINKEDIN_ACTOR_ID = "automation-lab/linkedin-jobs-scraper"


def stable_job_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def normalize_linkedin_job(item: Dict) -> Dict:
    """
    Normalize a raw LinkedIn job item from the Apify dataset
    into the shared Job-Bot schema.
    """
    title = item.get("title", "")
    location = item.get("location", "")
    
    is_remote = (
        bool(re.search(r"remote", title, re.IGNORECASE)) or 
        bool(re.search(r"remote", location, re.IGNORECASE)) or 
        item.get("workplaceType") == "Remote"
    )

    return {
        "source": "LinkedIn",
        "job_title": item.get("title", "Data Engineering"),
        "company": item.get("companyName", "Unknown"),
        "company_url": item.get("companyLinkedinUrl"),
        "company_logo": item.get("companyLogo"),
        "location": location or "United States",
        "remote": is_remote,
        "posted_date": item.get("postedAt"),
        "scraped_at": item.get("scrapedAt", datetime.utcnow().isoformat()),
        "url": item.get("url"),
        "apply_url": item.get("applyUrl") or item.get("url"),
        "description": item.get("descriptionText", ""),
        "description_html": item.get("descriptionHtml", ""),
        "requirements": None,
        "salary": item.get("salary"),
        "seniority_level": item.get("seniorityLevel"),
        "employment_type": item.get("employmentType"),
        "job_function": item.get("jobFunction"),
        "industries": item.get("industries"),
        "applicants_count": item.get("applicantsCount"),
        "benefits": item.get("benefits"),
        "id": stable_job_id("linkedin", str(item.get("id") or item.get("url") or item.get("title") or "")),
        "raw_payload": item,
    }

def fetch_linkedin_jobs_via_apify_api(options: Optional[Dict] = None) -> List[Dict]:
    """
    Fetch LinkedIn Data Engineering jobs via the Apify API.
    Uses the automation-lab/linkedin-jobs-scraper actor.
    """
    if options is None:
        options = {}

    try:
        run_input = {
            "searchQuery": options.get("searchQuery", config.JOB_KEYWORDS),
            "location": options.get("location", config.JOB_LOCATION),
            "maxJobs": options.get("maxJobs", 50),
            "workplaceType": options.get("workplaceType", "2"),  # Remote
            "datePosted": options.get("datePosted", "r86400"),  # Last 24 hours
            "sortBy": options.get("sortBy", "R"),              # Most recent
            "experienceLevel": options.get("experienceLevel", "all"),
            "jobType": options.get("jobType", "all"),
            "scrapeJobDetails": options.get("scrapeJobDetails", True),
            "maxRequestRetries": config.APIFY_MAX_RETRIES,
        }

        log(f"Starting LinkedIn Apify actor ({LINKEDIN_ACTOR_ID}) with input:")
        log(json.dumps(run_input, indent=2))

        run = client.actor(LINKEDIN_ACTOR_ID).call(
            run_input=run_input,
            run_timeout=timedelta(seconds=config.APIFY_TIMEOUT_SECONDS),
            memory_mbytes=256,
        )

        jobs = []
        dataset_id = getattr(run, "default_dataset_id", None) or (run.get("defaultDatasetId") if isinstance(run, dict) else None)
        if run and dataset_id:
            log(f"Actor run completed. Dataset ID: {dataset_id}")
            dataset_items = client.dataset(dataset_id).list_items().items

            for item in dataset_items:
                jobs.append(normalize_linkedin_job(item))
        else:
            error("Actor run completed but no dataset was returned.")

        log(f"LinkedIn API fetch completed: {len(jobs)} jobs found")
        return jobs

    except Exception as err:
        error(f"LinkedIn API fetch failed: {str(err)}")
        return []

def fetch_linkedin_jobs_from_dataset(dataset_id: str) -> List[Dict]:
    """
    Fetch jobs from a previous LinkedIn actor run's dataset.
    """
    try:
        log(f"Fetching jobs from existing dataset: {dataset_id}")
        dataset_items = client.dataset(dataset_id).list_items().items
        jobs = [normalize_linkedin_job(item) for item in dataset_items]
        
        log(f"Fetched {len(jobs)} jobs from dataset {dataset_id}")
        return jobs
    except Exception as err:
        error(f"Failed to fetch from dataset {dataset_id}: {str(err)}")
        return []
