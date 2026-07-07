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
    # Groq formats wait times as either plain seconds ("12.3s") or, once past a
    # minute, compound "1m5.664s" / "2h3m4s" — the plain-seconds-only pattern
    # silently failed to match the compound form, so we'd under-wait and burn
    # retries against a limit that hadn't cleared yet.
    match = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", str(exc), re.IGNORECASE)
    if match:
        hours = float(match.group(1)) if match.group(1) else 0.0
        minutes = float(match.group(2)) if match.group(2) else 0.0
        seconds = float(match.group(3)) if match.group(3) else 0.0
        return hours * 3600 + minutes * 60 + seconds + 0.5
    return None


def retry_groq_call(fn, max_retries: int = 5, base_delay: float = 2.0, rate_limit_base_delay: float = 6.0, fallback_fn=None):
    """
    Run `fn()` (a zero-arg callable that performs one Groq API call), retrying
    on failure. Rate-limit (429) errors get a longer, Retry-After-aware backoff
    than generic transient errors, since they need real cooldown time rather
    than a quick retry. When Groq doesn't give an exact wait time, the fallback
    schedule is capped rather than left to grow unbounded.

    If `fallback_fn` is given, a rate limit on `fn` triggers an immediate
    (no-wait) switch to `fallback_fn` instead of sleeping — this matters
    because Groq's per-model limits include a *daily* token cap (TPD), not
    just a per-minute one, and waiting out a near-exhausted daily cap can take
    many minutes. A fallback model draws from its own separate quota, so it
    clears the request right away instead of waiting on the primary model's
    cooldown.
    """
    last_exc = None
    current_fn = fn
    switched = False
    for attempt in range(max_retries):
        _wait_for_pacing_slot()
        try:
            return current_fn()
        except Exception as e:
            last_exc = e
            rate_limited = is_rate_limit_error(e)
            if rate_limited and fallback_fn is not None and not switched:
                print(f"[WARN] Groq rate limit (429) on primary model — switching to fallback model immediately: {e}")
                current_fn = fallback_fn
                switched = True
                continue
            if attempt == max_retries - 1:
                print(f"[ERR] Groq API failed after {max_retries} attempts: {e}")
                raise
            wait = extract_retry_after_seconds(e)
            if wait is None:
                wait = min(rate_limit_base_delay * (attempt + 1), 20.0) if rate_limited else base_delay * (2 ** attempt)
            kind = "rate limit (429)" if rate_limited else "error"
            print(f"[WARN] Groq API {kind} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise last_exc
