from pathlib import Path
import os
from dotenv import load_dotenv

# Explicitly load .env from the project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)


def _clear_blackhole_proxy_env() -> None:
    """
    Some environments inject a dummy proxy such as 127.0.0.1:9, which breaks
    Apify/API traffic. Remove only that placeholder so direct connections work.
    """
    proxy_keys = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "GIT_HTTP_PROXY",
        "GIT_HTTPS_PROXY",
    )

    for key in proxy_keys:
        value = os.environ.get(key)
        if value and "127.0.0.1:9" in value:
            os.environ.pop(key, None)


_clear_blackhole_proxy_env()

CANDIDATE_EMAIL = os.getenv("CANDIDATE_EMAIL", "candidate@example.com")
APIFY_TOKEN = os.getenv("APIFY_TOKEN")
JOB_KEYWORDS = os.getenv("JOB_KEYWORDS", "Data Engineer")
JOB_LOCATION = os.getenv("JOB_LOCATION", "United States")
REMOTE_ONLY = True
MAX_AGE_DAYS = 1
OUTPUT_DIR = Path("./data")

APIFY_MAX_RETRIES = int(os.getenv("APIFY_MAX_RETRIES", 3))
APIFY_TIMEOUT_SECONDS = int(os.getenv("APIFY_TIMEOUT_SECONDS", 300))
APIFY_MAX_CONCURRENCY = int(os.getenv("APIFY_MAX_CONCURRENCY", 5))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DICE_ACTOR_ID = os.getenv("DICE_ACTOR_ID", "shahidirfan/Dice-Job-Scraper")

# Kept as a separate token/client from APIFY_TOKEN above (even though it currently
# holds the same value) so a future rotation of either one can't silently break
# the other source's discovery.
BUILTIN_APIFY_TOKEN = os.getenv("BUILTIN_APIFY_TOKEN") or APIFY_TOKEN
BUILTIN_ACTOR_ID = os.getenv("BUILTIN_ACTOR_ID", "shahidirfan/BuiltIn-Jobs-Scraper")

# Kept separate from the shared APIFY_TIMEOUT_SECONDS (120s, works fine for
# LinkedIn/Indeed/Dice) — confirmed live that the Built In actor's own
# collectDetails:true per-job scraping needs longer than 120s for a full
# results_wanted:50 run and was hitting TIMED-OUT before returning any data.
# 300s matches the actor's own default run budget on Apify's side.
BUILTIN_APIFY_TIMEOUT_SECONDS = int(os.getenv("BUILTIN_APIFY_TIMEOUT_SECONDS", 300))

# Daily job report email (sent once per day at 12:00 AM Pakistan Standard Time)
REPORT_SMTP_EMAIL = os.getenv("REPORT_SMTP_EMAIL")
REPORT_SMTP_APP_PASSWORD = os.getenv("REPORT_SMTP_APP_PASSWORD")
REPORT_RECIPIENT_EMAIL = os.getenv("REPORT_RECIPIENT_EMAIL")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)
