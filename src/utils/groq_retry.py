"""
Shared retry/backoff + pacing helper for every Groq call site in the project
(the raw `Groq` client in optimize.py/cv_processor.py, and the `ChatGroq`
langchain wrapper in linkedin_apply.py/apply.py) so a 429 in any one of them
gets the same treatment: back off, honor Groq's own Retry-After hint when it
gives one, and retry — instead of failing the step outright.
"""
import re
import threading
import time

# Project-wide minimum gap between consecutive Groq requests, enforced across
# every call site via the shared lock below. This is a request-rate throttle,
# not just a per-call retry — it reduces how often we hit 429 in the first
# place when several LLM calls fire in quick succession (e.g. one per form
# step during an apply run).
MIN_INTERVAL_SECONDS = 1.5

_pacing_lock = threading.Lock()
_last_call_at = {"t": 0.0}


def _wait_for_pacing_slot():
    with _pacing_lock:
        now = time.time()
        wait = MIN_INTERVAL_SECONDS - (now - _last_call_at["t"])
        if wait > 0:
            time.sleep(wait)
        _last_call_at["t"] = time.time()


def is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None) if response is not None else None
    if status == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "rate_limit_exceeded" in text


def extract_retry_after_seconds(exc: Exception):
    """Read Groq's Retry-After header if the SDK exposes it, else parse it out of the error message."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
    match = re.search(r"try again in ([\d.]+)s", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 0.5
    return None


def retry_groq_call(fn, max_retries: int = 5, base_delay: float = 2.0, rate_limit_base_delay: float = 8.0):
    """
    Run `fn()` (a zero-arg callable that performs one Groq API call), retrying
    on failure. Rate-limit (429) errors get a longer, Retry-After-aware backoff
    than generic transient errors, since they need real cooldown time rather
    than a quick retry.
    """
    last_exc = None
    for attempt in range(max_retries):
        _wait_for_pacing_slot()
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt == max_retries - 1:
                print(f"[ERR] Groq API failed after {max_retries} attempts: {e}")
                raise
            rate_limited = is_rate_limit_error(e)
            wait = extract_retry_after_seconds(e)
            if wait is None:
                wait = rate_limit_base_delay * (attempt + 1) if rate_limited else base_delay * (2 ** attempt)
            kind = "rate limit (429)" if rate_limited else "error"
            print(f"[WARN] Groq API {kind} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise last_exc
