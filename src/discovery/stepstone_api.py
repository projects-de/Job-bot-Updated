import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from apify_client import ApifyClient
from src import config
from src.utils.logger import log, error
from src.utils.workplace_classifier import classify_workplace_type_from_title, classify_workplace_type_from_html

client = ApifyClient(config.APIFY_TOKEN)

# Public actor by shahidirfan (same publisher as our BuiltIn/Dice actors),
# confirmed live via a real 2-item test run. Scrapes StepStone.de only — the
# actor's own input schema has no location parameter ("currently not
# supported in Stepstone URLs, reserved for future use"), so every result is
# Germany-market by nature of the site, driven entirely by `keyword`.
STEPSTONE_ACTOR_ID = "shahidirfan/Stepstone-Job-Scraper"


def stable_job_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def normalize_stepstone_job(item: Dict) -> Dict:
    """
    Normalize a raw StepStone job item (confirmed live field names: title,
    company, location, date_posted, job_type, job_category, salary, url,
    description_html, description_text — no separate apply URL, postings
    link straight to the StepStone listing itself) into the shared
    Job-Bot schema.
    """
    title = item.get("title", "")
    description_html = item.get("description_html", "")

    workplace_type_enum = classify_workplace_type_from_title(title) or classify_workplace_type_from_html(description_html)
    is_remote = workplace_type_enum == "REMOTE"

    return {
        "source": "StepStone",
        # StepStone is always its own separate QA region in this codebase —
        # kept apart from the LinkedIn Dubai/Europe QA passes per explicit
        # instruction, not merged into either.
        "qa_region": "stepstone",
        "job_title": title or "QA Engineer",
        "company": item.get("company", "Unknown"),
        "location": item.get("location") or "Germany",
        "remote": is_remote,
        "workplace_type_enum": workplace_type_enum,
        "posted_date": item.get("date_posted"),
        "scraped_at": datetime.utcnow().isoformat(),
        "url": item.get("url"),
        "apply_url": item.get("url"),
        "description": item.get("description_text", ""),
        "description_html": description_html,
        "requirements": None,
        "salary": item.get("salary"),
        "employment_type": item.get("job_type"),
        # StepStone's own category label (e.g. "IT, IT-Testing") — kept as
        # extra metadata alongside our own job_category ("qa"), not used for
        # filtering since it's not consistently present across postings.
        "job_function": item.get("job_category"),
        "id": stable_job_id("stepstone", str(item.get("url") or title)),
        "raw_payload": item,
    }


def fetch_stepstone_jobs_via_apify_api(options: Optional[Dict] = None) -> List[Dict]:
    """
    Fetch jobs from StepStone.de via the Apify API.
    Uses the shahidirfan/Stepstone-Job-Scraper actor.
    """
    if options is None:
        options = {}

    try:
        run_input = {
            "keyword": options.get("keyword", config.STEPSTONE_QA_KEYWORD),
            "datePosted": options.get("datePosted", "last_7d"),
            "collectDetails": options.get("collectDetails", True),
            "maxItems": options.get("maxItems", 50),
            "maxPages": options.get("maxPages", 5),
            "dedupe": options.get("dedupe", True),
        }

        log(f"Starting StepStone Apify actor ({STEPSTONE_ACTOR_ID}) with input:")
        log(json.dumps(run_input, indent=2))

        run = client.actor(STEPSTONE_ACTOR_ID).call(
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
                jobs.append(normalize_stepstone_job(item))
        else:
            error("Actor run completed but no dataset was returned.")

        log(f"StepStone API fetch completed: {len(jobs)} jobs found")
        return jobs

    except Exception as err:
        error(f"StepStone API fetch failed: {str(err)}")
        return []
