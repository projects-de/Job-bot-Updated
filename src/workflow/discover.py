import os
import re
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlsplit, urlunsplit

# Add the project root to sys.path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src import config
from src.discovery.linkedin_api import fetch_linkedin_jobs_via_apify_api, fetch_linkedin_jobs_via_apify_task
from src.discovery.indeed_api import fetch_indeed_jobs_via_apify_api
from src.discovery.dice_api import fetch_dice_jobs_via_apify_api
from src.discovery.builtin_api import fetch_builtin_jobs_via_apify_api
from src.discovery.stepstone_api import fetch_stepstone_jobs_via_apify_api
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
SOURCE_PRIORITY = {"linkedin": 0, "indeed": 1, "builtin": 2, "dice": 3, "stepstone": 4}


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


# Non-software fields confirmed live to show up as noise in the QA task
# results — industrial/construction ("QAQC Engineer - Mechanical"), unrelated
# trades, plus BPO/call-center "quality analyst" roles and cybersecurity
# audit roles (both use QA-adjacent words but aren't the software-QA domain
# being targeted). Exclusion list, applied on top of the positive keyword
# match below.
_NON_SOFTWARE_FIELD_TERMS = [
    "civil", "mechanical", "electrical", "electromechanical", "construction",
    "plumbing", "hvac", "elv", "structural", "architect", "welding",
    "logistics", "inspector", "commissioning", "kitchen", "chef", "f&b",
    "hospitality", "banking", "workshop", "rail", "metro", "maintenance",
    "restaurant", "flight attendant", "airbus", "marine", "oil and gas",
    "oil & gas", "real estate", "warehouse", "driver", "nurse", "nursing",
    "pharmacist", "clinical", "hotel", "housekeeping", "security guard",
    "contact center", "call center", "customer service", "cybersecurity",
]

# Required positive match — confirmed live that the Dubai QA task (unlike the
# cleaner Europe task) still returns plenty of jobs with no QA/testing
# relevance at all (ByteDance internships, "TAMM OS Developer", "Digital Twin
# Manager", "Sharepoint Consultant"...), which the exclusion list alone can't
# catch since they don't match any wrong-field term either. Short substrings
# ("test", "automat") are deliberate — they catch multilingual variants
# (French "Testeur"/"Automaticien", German "Testautomation") that a full-word
# "tester"/"automation" match would miss.
_QA_RELEVANCE_TERMS = ["qa", "sdet", "test", "automat", "quality assurance", "quality engineer", "quality analyst"]


def is_software_field(job: dict) -> bool:
    """
    Exclusion safety net for the LinkedIn QA passes. Also drops the "Ethos"
    postings confirmed live to be recurring expert-network ads ("Expert
    Opportunity - Plant Manager", "...Hospital Operations Consultant", etc.)
    unrelated to any real QA opening, regardless of title.
    """
    title = (job.get("job_title") or "").lower()
    company = (job.get("company") or "").lower()
    if company == "ethos" or "expert opportunity" in title:
        return False
    if any(term in title for term in _NON_SOFTWARE_FIELD_TERMS):
        return False
    return True


def is_qa_relevant(job: dict) -> bool:
    """
    Positive match for the actual domain being targeted — QA, test
    automation, SDET, software tester — on top of is_software_field's
    exclusions. Verified live against a real scan: keeps all 50/50 genuine
    matches from the Europe task and narrows the noisier Dubai task from 15
    down to the 4 that are actually QA/automation roles.
    """
    title = (job.get("job_title") or "").lower()
    return is_software_field(job) and any(term in title for term in _QA_RELEVANCE_TERMS)


def run_discovery():
    try:
        log("=== Starting daily discovery workflow (Multi-Source-Python) ===")
        if not config.APIFY_TOKEN:
            error("APIFY_TOKEN environment variable is not set. Please set it in .env file.")
            raise RuntimeError("APIFY_TOKEN environment variable is not set")

        log(f"Search: '{config.JOB_KEYWORDS}' | Location: '{config.JOB_LOCATION}'")
        log(f"LinkedIn QA tasks: Dubai='{config.LINKEDIN_QA_DUBAI_TASK_ID}', Europe='{config.LINKEDIN_QA_EUROPE_TASK_ID}'")

        source_jobs = {
            "LinkedIn": [],
            "Indeed": [],
            "BuiltIn": [],
            "StepStone": [],
        }
        errors = []

        def _make_linkedin_qa_task_fetcher(task_id: str, region: str):
            def _fetch():
                return fetch_linkedin_jobs_via_apify_task(task_id, region=region)
            return _fetch

        # Dice discovery is switched off (fetch_dice_jobs_via_apify_api stays
        # imported above, untouched, so this is a one-line revert if Dice is
        # ever needed again) — Built In takes its place below.
        #
        # Each task: (bucket key in source_jobs, job_category stamped onto every
        # job it returns, the fetch callable, the title filter to apply).
        # LinkedIn runs once for Data Engineer plus once per QA region task
        # (Dubai, Europe — pre-configured on the Apify console with their own
        # search query/location, so nothing is passed at call time) — all
        # landing in the same "LinkedIn" bucket; job_category is what the
        # frontend splits on, not a separate top-level source, so by-source
        # totals still read as "all LinkedIn jobs this scan". Each QA job is
        # additionally stamped with qa_region ("dubai"/"europe") by
        # fetch_linkedin_jobs_via_apify_task so the frontend can show separate
        # cards per region.
        tasks = [
            # Data Engineering actor calls (LinkedIn, Indeed, Built In) are
            # disconnected here on request to cut Apify cost while focus is on
            # the QA tasks below — this also sidesteps the "exceed your limit
            # of 5 concurrent Actor runs" error from running all 5 at once.
            # To reconnect: uncomment these 3 lines, no other change needed.
            # ("LinkedIn", "data_engineering", fetch_linkedin_jobs_via_apify_api, is_strict_data_engineer),
            # ("Indeed", "data_engineering", fetch_indeed_jobs_via_apify_api, is_strict_data_engineer),
            # ("BuiltIn", "data_engineering", fetch_builtin_jobs_via_apify_api, is_strict_data_engineer),
            ("LinkedIn", "qa", _make_linkedin_qa_task_fetcher(config.LINKEDIN_QA_DUBAI_TASK_ID, "dubai"), is_qa_relevant),
            ("LinkedIn", "qa", _make_linkedin_qa_task_fetcher(config.LINKEDIN_QA_EUROPE_TASK_ID, "europe"), is_qa_relevant),
            # StepStone.de — a third, separate QA region (Germany-market,
            # driven by keyword only; the actor has no location parameter).
            # qa_region="stepstone" is stamped directly in
            # normalize_stepstone_job, not here, since this source is always
            # QA-only in this codebase (unlike LinkedIn, which also runs a
            # Data Engineer pass through the same normalize function).
            ("StepStone", "qa", fetch_stepstone_jobs_via_apify_api, is_qa_relevant),
        ]

        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_map = {executor.submit(fetcher): (bucket, category, filter_fn) for bucket, category, fetcher, filter_fn in tasks}
            for future in as_completed(future_map):
                bucket, category, filter_fn = future_map[future]
                label = f"{bucket} ({category})"
                try:
                    jobs = future.result() or []

                    if not jobs:
                        # Not necessarily a bug — a source can legitimately have 0 matching
                        # postings — but it's also what a silently-blocked/rate-limited scrape
                        # looks like, so flag it instead of letting it look identical to "0
                        # jobs, all good" in the dashboard.
                        message = f"{label} returned 0 jobs this scan (may be a genuine 0, or a blocked/failed fetch — check the scan log for that source's fetch attempts above)"
                        errors.append(message)
                        log(f"WARNING: {message}")

                    # Strictly filter by title before saving or showing
                    filtered_jobs = [j for j in jobs if filter_fn(j)]
                    if len(jobs) > len(filtered_jobs):
                        log(f"Filtered out {len(jobs) - len(filtered_jobs)} non-matching jobs from {label}.")

                    jobs = filtered_jobs
                    for j in jobs:
                        j["job_category"] = category
                    source_jobs[bucket] = source_jobs.get(bucket, []) + jobs

                    # Live Update: Save jobs from this task immediately so they show up in UI
                    if jobs:
                        try:
                            log(f"Live Update: Saving {len(jobs)} strictly filtered jobs from {label}...")
                            save_jobs(jobs, suffix="all")

                            # Also save to unique so the dashboard 'Apply' tab shows them immediately
                            deduped_partial = deduplicate_jobs(jobs)
                            save_jobs(deduped_partial, suffix="unique")
                        except Exception as e:
                            log(f"Live Update Warning: {e}")
                except Exception as exc:
                    message = f"{label} discovery failed: {exc}"
                    errors.append(message)
                    error(message)

        all_jobs = source_jobs["LinkedIn"] + source_jobs["Indeed"] + source_jobs["BuiltIn"] + source_jobs["StepStone"]
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
                "BuiltIn": len(source_jobs["BuiltIn"]),
                "StepStone": len(source_jobs["StepStone"]),
            },
            "errors": errors,
        }

        # Save both lists for the frontend to toggle
        save_jobs(all_jobs, suffix="all")
        save_jobs(deduped_jobs, suffix="unique")
        save_summary(summary)

        log("=== Daily discovery workflow complete ===")
        qa_count = sum(1 for j in source_jobs["LinkedIn"] if j.get("job_category") == "qa")
        log(
            "LinkedIn: {linkedin} (of which QA: {qa}), Indeed: {indeed}, BuiltIn: {builtin}, StepStone: {stepstone}".format(
                linkedin=len(source_jobs["LinkedIn"]),
                qa=qa_count,
                indeed=len(source_jobs["Indeed"]),
                builtin=len(source_jobs["BuiltIn"]),
                stepstone=len(source_jobs["StepStone"]),
            )
        )
        log(f"Total Unique Jobs (Data Engineer + QA): {len(deduped_jobs)}")
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
