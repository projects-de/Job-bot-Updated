import hashlib
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from apify_client import ApifyClient
from src import config
from src.utils.logger import log, error
from src.utils.workplace_classifier import classify_workplace_type_from_title, classify_workplace_type_from_html

client = ApifyClient(config.APIFY_TOKEN)

# Actor ID for the LinkedIn Jobs Scraper by automation-lab
LINKEDIN_ACTOR_ID = "automation-lab/linkedin-jobs-scraper"


def stable_job_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _extract_workplace_type(
    description_html: str, title: str, location: str, requested_workplace_type: str = ""
) -> str:
    """
    Extract workplace type (REMOTE, HYBRID, IN_OFFICE) for a LinkedIn job.
    The actual text-parsing rules (title check, labeled lines, day-counts,
    explicit phrasing, header line) live in the shared
    src/utils/workplace_classifier module so the same tuned logic can also
    run against an external company careers page's text — LinkedIn's own
    page sometimes has no signal at all even though the employer's own ATS
    page states it (confirmed live). This function adds the one rule that's
    specific to how LinkedIn's own search works: if nothing in the title or
    description matched AND this job came from a search that was itself
    filtered to Remote-only on LinkedIn's own side (requested_workplace_type
    == "2", e.g. the Data Engineer pass), default to REMOTE — confirmed live
    that several such jobs ("Staff Data Engineer" at Overstory, "Data
    Engineer II") have no textual remote statement at all, yet are
    guaranteed remote by the search filter itself. Never applied to searches
    that aren't remote-scoped (e.g. the QA tasks, which return a genuine mix
    of types), since that would wrongly stamp REMOTE on real Hybrid/On-site
    results. Returns "" if nothing matched — deliberately conservative:
    showing a job in no Remote/Hybrid/Onsite card is far less harmful than
    showing it in the wrong one.
    """
    title_type = classify_workplace_type_from_title(title)
    if title_type:
        return title_type

    html_type = classify_workplace_type_from_html(description_html)
    if html_type:
        return html_type

    if requested_workplace_type == "2":
        return "REMOTE"

    return ""


def normalize_linkedin_job(item: Dict, requested_workplace_type: str = "") -> Dict:
    """
    Normalize a raw LinkedIn job item from the Apify dataset
    into the shared Job-Bot schema.
    """
    title = item.get("title", "")
    location = item.get("location", "")
    description_html = item.get("descriptionHtml", "")

    workplace_type_enum = _extract_workplace_type(description_html, title, location, requested_workplace_type)
    is_remote = workplace_type_enum == "REMOTE"

    return {
        "source": "LinkedIn",
        "job_title": item.get("title", "Data Engineering"),
        "company": item.get("companyName", "Unknown"),
        "company_url": item.get("companyLinkedinUrl"),
        "company_logo": item.get("companyLogo"),
        "location": location or "United States",
        "remote": is_remote,
        "workplace_type_enum": workplace_type_enum,
        "posted_date": item.get("postedAt"),
        "scraped_at": item.get("scrapedAt", datetime.utcnow().isoformat()),
        "url": item.get("url"),
        "apply_url": item.get("applyUrl") or item.get("url"),
        "description": item.get("descriptionText", ""),
        "description_html": description_html,
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
            "maxJobs": options.get("maxJobs", 80),
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
                jobs.append(normalize_linkedin_job(item, requested_workplace_type=run_input["workplaceType"]))
        else:
            error("Actor run completed but no dataset was returned.")

        log(f"LinkedIn API fetch completed: {len(jobs)} jobs found")
        return jobs

    except Exception as err:
        error(f"LinkedIn API fetch failed: {str(err)}")
        return []

def fetch_linkedin_jobs_via_apify_task(task_id: str, region: str = "") -> List[Dict]:
    """
    Fetch LinkedIn jobs via a pre-configured Apify Task instead of the raw
    actor: search query, location, and other filters live in the task's own
    saved input on the Apify console. The one field overridden here —
    scrapeJobDetails — is merged on top of that saved input for this run
    only (Apify Tasks merge a partial task_input over the stored config, not
    replace it), everything else stays exactly as configured on the console.
    It's required: confirmed live that without it, descriptionHtml/
    workplaceType come back empty for every job, so Remote/Hybrid/Onsite
    classification (_extract_workplace_type) silently fails for the whole
    QA tab regardless of region. Used for the QA passes (one task per
    region, e.g. Dubai/Europe); `region` is stamped onto each returned job
    so the frontend can split them into separate location cards.
    """
    try:
        log(f"Starting LinkedIn Apify task '{task_id}' (region={region or 'unspecified'})")

        run = client.task(task_id).call(
            task_input={"scrapeJobDetails": True},
            run_timeout=timedelta(seconds=config.APIFY_TIMEOUT_SECONDS),
            memory_mbytes=256,
        )

        jobs = []
        dataset_id = getattr(run, "default_dataset_id", None) or (run.get("defaultDatasetId") if isinstance(run, dict) else None)
        if run and dataset_id:
            log(f"Task run completed. Dataset ID: {dataset_id}")
            dataset_items = client.dataset(dataset_id).list_items().items

            for item in dataset_items:
                job = normalize_linkedin_job(item)
                if region:
                    job["qa_region"] = region
                jobs.append(job)
        else:
            error(f"Task '{task_id}' run completed but no dataset was returned.")

        log(f"LinkedIn task fetch completed ({task_id}): {len(jobs)} jobs found")
        return jobs

    except Exception as err:
        error(f"LinkedIn task fetch failed ({task_id}): {str(err)}")
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
