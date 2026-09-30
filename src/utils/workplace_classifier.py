"""
Shared Remote/Hybrid/On-site text classifier. Originally built for LinkedIn's
own job descriptions (src/discovery/linkedin_api.py) and factored out here so
the exact same, live-data-tuned rules can also run against an external
company careers page's text (src/discovery/linkedin_classify.py) when
LinkedIn's own page has no signal — the real "Remote"/"Hybrid" statement
sometimes only lives on the employer's own ATS page, not on LinkedIn.

Deliberately conservative throughout: only concrete, role-specific signals
count (a labeled line, a real day-count schedule, explicit phrasing, or a
title/header statement). A bare keyword appearing anywhere in the page is NOT
enough — confirmed live that free text routinely mentions multiple
work-arrangement words in unrelated contexts (generic benefits blurbs, vague
policy statements, unrelated leave-allowance numbers near the keyword).
Returns "" when no concrete signal is found rather than guessing — showing a
job in no Remote/Hybrid/Onsite card is far less harmful than the wrong one.
"""
import re

_WORKPLACE_LABEL_RE = re.compile(
    r"(work\s*arrangement|work\s*model|work\s*mode|mode\s*of\s*work|"
    r"workplace\s*type|job\s*type|work\s*type|work\s*setup)\s*[:\-]\s*(.+)"
)
_WORKPLACE_TYPE_RE = r"(hybrid|remote|on.?site|in.?office|office.?based)"
# Digits and small spelled-out numbers both count ("hybrid... 2 days a week"
# and "hybrid... three days a week" are equally concrete statements).
_WORKPLACE_DAYCOUNT_RE = re.compile(
    r"(?:\d+|one|two|three|four|five|six|seven)\s*days?\s*(?:a\s*week|per\s*week|/\s*week|"
    r"in\s*(?:the\s*)?office|office|from\s*home|at\s*home|onsite|on[\s\-]?site|remote)"
)
# Checked in this order (not left-to-right through the text) when matching a
# type word against a nearby day-count: confirmed live that a Hybrid
# description routinely names both its components ("work onsite in a hybrid
# model at a minimum of 3 days per week", "combines remote work with the
# benefits of in-office engagement... two days a week") — the component word
# (onsite/remote) can sit textually closer to the day-count detail than the
# word "hybrid" itself, so a naive first-match-wins scan picks the wrong,
# more specific-looking word. Checking every "hybrid" occurrence in the whole
# text first avoids that regardless of which component happens to be nearer.
_WORKPLACE_DAYCOUNT_PRIORITY = (
    ("hybrid", "HYBRID"),
    ("remote", "REMOTE"),
    (r"on.?site|in.?office|office.?based", "IN_OFFICE"),
)
_WORKPLACE_DAYCOUNT_WINDOW = 200


def workplace_type_of(word: str) -> str:
    if "hybrid" in word:
        return "HYBRID"
    if "remote" in word:
        return "REMOTE"
    return "IN_OFFICE"


def classify_workplace_type_from_title(title: str) -> str:
    """
    A job title stating a type directly ("Senior Data Engineer (Remote)",
    "Data Engineer - Remote") is a deliberate, reliable recruiter-written
    tag — checked before any page text at all.
    """
    title_match = re.search(_WORKPLACE_TYPE_RE, (title or "").lower())
    if title_match:
        return workplace_type_of(title_match.group(1))
    return ""


def classify_workplace_type_from_html(html: str) -> str:
    """
    Parse an arbitrary HTML page's text (a LinkedIn description or an
    external careers-page body) for a concrete, role-specific work-type
    statement. See module docstring for the full rule rationale.
    """
    if not html:
        return ""

    lowered = html.lower()
    # Turn block-level boundaries into newlines BEFORE stripping tags, so a
    # 'label: value' pair stays isolated on its own line instead of bleeding
    # into the next paragraph's label once tags are gone.
    text = re.sub(r"</(p|li|div|h[1-6])\s*>", "\n", lowered)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.split("\n") if ln.strip()]
    flat = " ".join(lines)

    for line in lines:
        m = _WORKPLACE_LABEL_RE.search(line)
        if m:
            type_m = re.search(_WORKPLACE_TYPE_RE, m.group(2)[:60])
            if type_m:
                return workplace_type_of(type_m.group(1))

    for pattern, label in _WORKPLACE_DAYCOUNT_PRIORITY:
        for m in re.finditer(pattern, flat):
            start, end = m.span()
            window = flat[max(0, start - _WORKPLACE_DAYCOUNT_WINDOW):end + _WORKPLACE_DAYCOUNT_WINDOW]
            if _WORKPLACE_DAYCOUNT_RE.search(window):
                return label

    for m in re.finditer(r"(?:position is a|is a)\s+" + _WORKPLACE_TYPE_RE, flat):
        return workplace_type_of(m.group(1))
    for m in re.finditer(_WORKPLACE_TYPE_RE + r"\s+working (?:opportunity|model) (?:based (?:in|from))", flat):
        return workplace_type_of(m.group(1))

    head_match = re.search(_WORKPLACE_TYPE_RE, flat[:150])
    if head_match:
        return workplace_type_of(head_match.group(1))

    return ""
