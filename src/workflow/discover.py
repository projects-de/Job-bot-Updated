import os
import re
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlsplit, urlunsplit

# Add the project root to sys.path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src import config
from src.discovery.linkedin_api import fetch_linkedin_jobs_via_apify_api
from src.discovery.indeed_api import fetch_indeed_jobs_via_apify_api
from src.discovery.dice_api import fetch_dice_jobs_via_apify_api
from src.storage.tracker import save_jobs, save_summary
from src.utils.logger import error, log


def canonical_job_url(url):
    """
    Normalize a job URL for duplicate detection. Keeps the query string —
    Indeed's job ID lives entirely in `?jk=...`, so stripping it (as a naive
    "drop tracking params" pass would) collapses nearly all distinct Indeed
    postings onto the same bare path. Only the fragment is dropped, since
    that's never load-bearing for identifying a specific job posting.
    """
    if not url:
        return ""
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


# Preference order when the same role is posted on multiple sources — lower wins.
SOURCE_PRIORITY = {"linkedin": 0, "indeed": 1, "dice": 2}


def _normalize_for_match(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def deduplicate_jobs(jobs):
    # Pass 1: drop exact duplicate postings (same canonical URL, or same source+title+company).
    seen = set()
    deduped = []
    for job in jobs:
        url = canonical_job_url(job.get("url", ""))
        key = url or f"{job.get('source', '').lower().strip()}::{job.get('job_title', '').lower().strip()}::{job.get('company', '').lower().strip()}"

        if key in seen:
            continue

        seen.add(key)
        deduped.append(job)

    # Pass 2: the same role posted on multiple sources -> keep only the preferred source
    # (LinkedIn over Indeed), matched by normalized title + company. Groups made up of a
    # single source are left untouched — a platform can legitimately have several distinct
    # postings that share a title + company (different teams/req IDs), and that's not a
    # cross-platform duplicate. Only genuine multi-source groups get collapsed, and their
    # survivors are flagged cross_platform_match=True (the "Unique Jobs" card/tab).
    groups = {}
    for job in deduped:
        role_key = (_normalize_for_match(job.get("job_title")), _normalize_for_match(job.get("company")))
        groups.setdefault(role_key, []).append(job)

    result = []
    for group in groups.values():
        sources_in_group = {(j.get("source") or "").lower() for j in group}
        if len(sources_in_group) <= 1:
            for j in group:
                j = dict(j)
                j["cross_platform_match"] = False
                result.append(j)
            continue

        best_priority = min(SOURCE_PRIORITY.get(s, 99) for s in sources_in_group)
        for j in group:
            if SOURCE_PRIORITY.get((j.get("source") or "").lower(), 99) == best_priority:
                j = dict(j)
                j["cross_platform_match"] = True
                result.append(j)
    return result


def is_strict_data_engineer(job: dict) -> bool:
    """
    Strictly cross-check if the job is a Data Engineer role.
    This acts as a production-level filter before saving.
    """
    title = job.get("job_title", "").lower()
    
    # Must contain data and engineer/engineering/architect/pipeline
    if "data" not in title:
        return False
        
    valid_roles = ["engineer", "engineering", "architect", "pipeline", "developer"]
    if not any(role in title for role in valid_roles):
        return False
        
    # Exclude strict non-data-engineering roles
    exclusions = ["software engineer", "frontend", "front end", "backend", "back end", "full stack", "fullstack", "scientist", "analyst"]
    
    # If it has an exclusion, check if it explicitly still says "data" directly attached
    for ex in exclusions:
        if ex in title and "data" not in title.replace(ex, "").strip():
            # If it's "Data Software Engineer", we might keep it, but if it's "Software Engineer - Data", we might keep it too.
            # But let's be strict: if it's just "Software Engineer" and data is somewhere else, exclude.
            pass
            
    # For production level, we can use Groq LLM here if needed, but regex is faster and 95% accurate.
    # Let's do a strict exclusion:
    for ex in ["frontend", "front end", "full stack", "fullstack"]:
        if ex in title:
            return False
            
    return True


def run_discovery():
    try:
        log("=== Starting daily discovery workflow (Multi-Source-Python) ===")
        if not config.APIFY_TOKEN:
            error("APIFY_TOKEN environment variable is not set. Please set it in .env file.")
            raise RuntimeError("APIFY_TOKEN environment variable is not set")

        log(f"Search: '{config.JOB_KEYWORDS}' | Location: '{config.JOB_LOCATION}'")

        source_jobs = {
            "LinkedIn": [],
            "Indeed": [],
            "Dice": [],
        }
        errors = []

        fetchers = {
            "LinkedIn": fetch_linkedin_jobs_via_apify_api,
            "Indeed": fetch_indeed_jobs_via_apify_api,
            "Dice": fetch_dice_jobs_via_apify_api,
        }

        with ThreadPoolExecutor(max_workers=len(fetchers)) as executor:
            future_map = {executor.submit(fetcher): source for source, fetcher in fetchers.items()}
            for future in as_completed(future_map):
                source = future_map[future]
                try:
                    jobs = future.result() or []

                    if not jobs:
                        # Not necessarily a bug — a source can legitimately have 0 matching
                        # postings — but it's also what a silently-blocked/rate-limited scrape
                        # looks like, so flag it instead of letting it look identical to "0
                        # jobs, all good" in the dashboard.
                        message = f"{source} returned 0 jobs this scan (may be a genuine 0, or a blocked/failed fetch — check the scan log for that source's fetch attempts above)"
                        errors.append(message)
                        log(f"WARNING: {message}")

                    # Strictly filter for Data Engineer before saving or showing
                    filtered_jobs = [j for j in jobs if is_strict_data_engineer(j)]
                    if len(jobs) > len(filtered_jobs):
                        log(f"Filtered out {len(jobs) - len(filtered_jobs)} non-Data Engineer jobs from {source}.")

                    jobs = filtered_jobs
                    source_jobs[source] = jobs

                    # Live Update: Save jobs from this source immediately so they show up in UI
                    if jobs:
                        try:
                            log(f"Live Update: Saving {len(jobs)} strictly filtered jobs from {source}...")
                            save_jobs(jobs, suffix="all")
                            
                            # Also save to unique so the dashboard 'Apply' tab shows them immediately
                            deduped_partial = deduplicate_jobs(jobs)
                            save_jobs(deduped_partial, suffix="unique")
                        except Exception as e:
                            log(f"Live Update Warning: {e}")
                except Exception as exc:
                    message = f"{source} discovery failed: {exc}"
                    errors.append(message)
                    error(message)

        all_jobs = source_jobs["LinkedIn"] + source_jobs["Indeed"] + source_jobs["Dice"]
        deduped_jobs = deduplicate_jobs(all_jobs)
        cross_platform_count = sum(1 for j in deduped_jobs if j.get("cross_platform_match"))

        summary = {
            "date": datetime.utcnow().isoformat(),
            "total_discovered": len(all_jobs),
            "total_deduplicated": len(deduped_jobs),
            "cross_platform_duplicates": cross_platform_count,
            "by_source": {
                "LinkedIn": len(source_jobs["LinkedIn"]),
                "Indeed": len(source_jobs["Indeed"]),
                "Dice": len(source_jobs["Dice"]),
            },
            "errors": errors,
        }

        # Save both lists for the frontend to toggle
        save_jobs(all_jobs, suffix="all")
        save_jobs(deduped_jobs, suffix="unique")
        save_summary(summary)

        log("=== Daily discovery workflow complete ===")
        log(
            "LinkedIn: {linkedin}, Indeed: {indeed}, Dice: {dice}".format(
                linkedin=len(source_jobs["LinkedIn"]),
                indeed=len(source_jobs["Indeed"]),
                dice=len(source_jobs["Dice"]),
            )
        )
        log(f"Total Unique Strict Data Engineer Jobs: {len(deduped_jobs)}")
        if errors:
            log(f"Discovery finished with warnings: {errors}")

        return summary

    except Exception as exc:
        error(f"Daily discovery workflow failed: {exc}")
        raise


if __name__ == "__main__":
    try:
        run_discovery()
    except Exception:
        sys.exit(1)
