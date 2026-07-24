import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from apify_client import ApifyClient
from src import config
from src.utils.logger import log, error

client = ApifyClient(config.BUILTIN_APIFY_TOKEN)

def stable_job_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"

_US_COUNTRY_NAMES = ("USA", "US", "UNITED STATES", "UNITED STATES OF AMERICA")


def _is_us_location(item: Dict) -> bool:
    """
    True only if the job's actual location requirement is the United States —
    deliberately NOT parsed from the free-text `location` field, which for a
    remote posting is often just the company's nearest anchor office address
    (e.g. "Austin, Texas, USA" on a job that's actually open to remote
    candidates across multiple countries) and would silently misclassify it.
    Prefers the structured fields the Apify actor already provides:
    `applicant_location_requirements` (the actual list of countries a remote
    candidate must be based in) first, then each `job_locations[].country`,
    and only falls back to scanning the free-text `location` string if neither
    structured field is present.
    """
    reqs = item.get("applicant_location_requirements") or []
    if reqs:
        return any(str(r).strip().upper() in _US_COUNTRY_NAMES for r in reqs)

    job_locations = item.get("job_locations") or []
    countries = [
        str(jl.get("country") or "").strip().upper()
        for jl in job_locations if isinstance(jl, dict)
    ]
    if countries:
        return any(c in _US_COUNTRY_NAMES for c in countries)

    loc_text = (item.get("location") or "").upper()
    return bool(re.search(r"\b(USA|US|UNITED STATES)\b", loc_text))


def normalize_builtin_job(item: Dict) -> Dict:
    """
    Normalize a raw Built In job item from the "shahidirfan/BuiltIn-Jobs-Scraper"
    Apify actor's dataset into the shared Job-Bot schema. Confirmed live against
    real actor output: `apply_url` is the actual employer application destination
    (e.g. a company's own ATS), separate from `url` (the builtin.com listing page)
    — unlike Dice, there's no click-through needed to resolve it.
    """
    title = item.get("title") or ""
    location = item.get("location") or ""
    workplace_type = (item.get("workplace_type") or "").lower()
    workplace_type_enum = (item.get("workplace_type_enum") or "").upper()

    # Strictly enum-based — confirmed live on a real "Principal Data Engineer"
    # posting at Upside: workplace_type is literally the string "Remote or
    # Hybrid" for a job whose workplace_type_enum is "HYBRID", so a naive
    # substring check ("remote" in workplace_type) wrongly classified a hybrid
    # job as remote. Only the actor's own REMOTE/HYBRID/IN_OFFICE enum is
    # trusted now, not the display text or a keyword scan of the title/location
    # (both of which can mention "remote" in an unrelated sense, e.g. "Remote
    # Sensing" job titles, or an office located in a town literally named
    # Remote).
    is_remote = workplace_type_enum == "REMOTE"

    # Truly-remote AND explicitly open to US-based candidates — the fields
    # backing the dedicated "Built In Remote Jobs" dashboard card. See
    # _is_us_location() for why this can't just regex the `location` string.
    is_remote_us = is_remote and _is_us_location(item)

    description = item.get("description_text") or ""
    description_html = item.get("description_html") or ""

    listing_url = item.get("url") or ""
    apply_url = item.get("apply_url") or listing_url
    raw_id = item.get("job_id") or item.get("drupal_id") or listing_url or title

    salary = item.get("salary_range_short")
    if not salary and (item.get("salary_json_min") or item.get("salary_json_max")):
        lo = item.get("salary_json_min")
        hi = item.get("salary_json_max")
        currency = item.get("salary_json_currency") or "USD"
        unit = item.get("salary_json_unit") or ""
        salary = f"{lo or ''}-{hi or ''} {currency} {unit}".strip()

    return {
        "source": "BuiltIn",
        "job_title": title or "Data Engineering",
        "company": item.get("company") or "Unknown",
        "company_url": item.get("company_url"),
        "company_logo": item.get("company_logo"),
        "location": location or "United States",
        "remote": is_remote,
        "workplace_type_enum": workplace_type_enum,
        "is_remote_us": is_remote_us,
        "posted_date": item.get("date_posted") or item.get("posted_text"),
        "scraped_at": datetime.utcnow().isoformat(),
        "url": listing_url,
        "apply_url": apply_url,
        "description": description,
        "description_html": description_html,
        "requirements": None,
        "salary": salary,
        "employment_type": item.get("employment_type"),
        "id": stable_job_id("builtin", str(raw_id)),
        "raw_payload": item,
    }

def fetch_builtin_jobs_via_apify_api(options: Optional[Dict] = None) -> List[Dict]:
    """
    Fetch Built In (builtin.com) jobs via the "shahidirfan/BuiltIn-Jobs-Scraper" Apify actor.
    """
    if options is None:
        options = {}

    try:
        run_input = {
            "keyword": options.get("query", config.JOB_KEYWORDS),
            "location": options.get("location", config.JOB_LOCATION),
            "results_wanted": options.get("results_wanted", 50),
            "proxyConfiguration": {"useApifyProxy": True},
        }

        log(f"Starting Built In Apify actor ({config.BUILTIN_ACTOR_ID}) with input:")
        log(json.dumps(run_input, indent=2))

        run = client.actor(config.BUILTIN_ACTOR_ID).call(
            run_input=run_input,
            run_timeout=timedelta(seconds=config.BUILTIN_APIFY_TIMEOUT_SECONDS),
        )

        dataset_id = getattr(run, "default_dataset_id", None) or (run.get("defaultDatasetId") if isinstance(run, dict) else None)

        jobs = []
        if dataset_id:
            log(f"Actor run completed. Dataset ID: {dataset_id}")
            dataset_items = client.dataset(dataset_id).list_items().items

            for item in dataset_items:
                jobs.append(normalize_builtin_job(item))
        else:
            error("Actor run completed but no dataset was returned.")

        log(f"Built In API fetch completed: {len(jobs)} jobs found")
        return jobs

    except Exception as err:
        error(f"Built In API fetch failed: {str(err)}")
        return []
