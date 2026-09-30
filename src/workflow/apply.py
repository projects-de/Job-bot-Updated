import json
import os
import re
import time
from urllib.parse import urlparse
from src.utils.step_logger import log_step
import threading
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from src.config import USER_AGENT, CANDIDATE_EMAIL, GROQ_API_KEY
from src.utils.groq_retry import retry_groq_call
from src.storage.tracker import log_application, save_tailored_application, save_job_snapshot, JobStatus, APPLICATIONS_DIR, UPLOADS_DIR
from src.workflow.optimize import render_tailored_cv_latex, _lean_cv_for_prompt
from src.workflow.indeed_apply import IndeedApplier, _is_indeed_url
from src.workflow.linkedin_apply import run_linkedin_agent
import glob

# Global events for human-in-the-loop review
review_event = threading.Event()
skip_event = threading.Event()

# Sets an <input>/<textarea>'s value through the element's native property setter
# (bypassing React/Vue's overridden setter) and fires input/change events, so
# framework-controlled fields actually register the change instead of silently
# keeping their old content — plain `.value = x` or Ctrl+A/Backspace alone can
# fail to clear these, leaving newly-typed text appended after stale content.
_NATIVE_SET_VALUE_JS = """
    const el = arguments[0];
    const value = arguments[1];
    const proto = el.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
    const descriptor = Object.getOwnPropertyDescriptor(proto, 'value');
    if (descriptor && descriptor.set) {
        descriptor.set.call(el, value);
    } else {
        el.value = value;
    }
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
"""

REVIEW_TIMEOUT_SECONDS = 180

# Used to resolve a state abbreviation parsed out of a CV's free-text "City, ST"
# location string into the full name a <select> option might use instead (or
# vice versa) — some ATS state dropdowns list "TX", others list "Texas".
US_STATE_ABBR_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia",
}

# Gender/EEO "voluntary self-identification" fields are almost always legally optional —
# a CV has no factual basis to answer them, so the safe, consistent behavior is to select
# whichever option means "I choose not to answer" rather than ever guessing.
DECLINE_TO_ANSWER_KEYWORDS = ("decline", "prefer not", "do not wish", "don't wish", "not to say", "not disclose")

# Epoch time the current review-wait period expires, or None when no job is
# currently waiting on human review. A plain dict (not a module-level float) so
# the API layer can import and read it live without a getter function.
review_deadline_epoch = {"value": None}

# True only once a job has actually reached "REVIEW REQUIRED" — i.e. the
# bot's own fill work (including any native agent's internal retry/wait
# loops) is done and it's genuinely waiting on a Skip/Submit click. Distinct
# from application_status["running"] (True the whole time a job is being
# processed, including while the bot is still actively filling/retrying) —
# confirmed live on a QTech US Inc job stuck for minutes inside Dice's own
# retry loop: the dashboard's Skip/Submit banner and countdown were shown
# and clickable the entire time even though nothing was listening for them
# yet, which made a Skip click during that phase silently do nothing. The
# API layer reads this the same way it already reads review_deadline_epoch.
awaiting_review = {"value": False}


class SkipRequested(Exception):
    """
    Raised from inside a native agent's own long-running wait loops (Dice's
    or LinkedIn's _wait_for_human_to_advance_step/_wait_for_password_entry,
    LinkedIn's node_wait_login) the moment skip_event is set, so a Skip click
    takes effect immediately even while the bot is still stuck mid-fill —
    not just once it reaches REVIEW REQUIRED. Explicitly caught in
    apply_to_job() and handled exactly like a skip requested during the
    normal review wait (SKIPPED status, tabs closed, back to overview) —
    caught before the generic `except Exception` there, which would
    otherwise mislabel it as a FAILED job.
    """
    pass


class BrowserCrashed(Exception):
    """
    Raised when the entire ChromeDriver/browser process is gone — not just
    one tab closed (that's the "no such window" case _recover_dead_window()
    already handles by switching to a surviving tab). Confirmed live on a
    Tata Consultancy Services job: every Selenium call started failing with
    a connection-refused error (the local ChromeDriver HTTP server itself
    was unreachable), and with nothing distinguishing this from an ordinary
    per-selector error, the bot spent 11+ minutes blindly retrying one field
    selector after another against a connection that could never come back.
    Checked for via check_browser_alive() at the same points SkipRequested
    is checked; caught in apply_to_job(), which relaunches a fresh browser
    and retries the job once before giving up.
    """
    pass


class BrowserClosedByUser(Exception):
    """
    Raised when the browser window is confirmed genuinely gone — zero window
    handles left, but the ChromeDriver connection itself is still fine (see
    BrowserCrashed for the separate whole-process-dead case) — while waiting
    for you to manually complete a step the bot paused on because it
    couldn't fill it itself.

    Confirmed live this exact gap existed: the wait loops for that pause
    (_wait_for_human_to_advance_step() in both linkedin_apply.py and
    builtin_apply.py) only ever checked for the whole-process-dead case via
    check_browser_alive() — never "the window itself is just gone" — so
    when you closed the browser mid-manual-fill, one of two silent failures
    followed depending on which file: linkedin_apply.py's version inferred
    "you finished the step" from the page-read failing (its page-signature
    and validation-error checks both fail safe to ""/[] on any error, which
    looks identical to "the page changed and has no errors"), so it resumed
    filling against a dead browser and eventually reached a false "form
    fully filled" state that got marked Submitted despite nothing real ever
    being filled or submitted. builtin_apply.py's version instead looped
    forever with no way out, since its dead-tab recovery correctly returns
    False when no tabs remain but nothing then broke out of the loop.

    Deliberately NOT retried like BrowserCrashed — relaunching and refilling
    from scratch would just hit the exact same "needs your help" wall again,
    since nothing about the field the bot couldn't handle has changed.
    Caught in apply_to_job() and treated as an honest "not submitted" stop —
    same JobStatus.TAILORED / "User closed browser before submit" outcome
    the review-wait loop already produces when the browser closes there, so
    the job lands back in today's Application Pipeline either way.
    """
    pass


_CONNECTION_LOST_SIGNATURES = (
    "actively refused",
    "failed to establish a new connection",
    "no connection could be made",
)


def check_browser_alive(driver):
    """
    Distinguishes "the whole browser/ChromeDriver process is gone" from "one
    tab closed" — the former can never be recovered by switching windows, so
    it needs a completely different response (relaunch), not the existing
    per-tab recovery logic. A cheap, low-cost driver call (just reading the
    window handle list) is enough to surface the failure without adding
    meaningful overhead to the loops that already call this every 2 seconds.
    """
    try:
        driver.window_handles
    except Exception as e:
        msg = str(e).lower()
        if any(sig in msg for sig in _CONNECTION_LOST_SIGNATURES):
            raise BrowserCrashed(str(e))


def _load_tailored_for_job(job_id):
    """Load the pre-generated tailored CV and cover letter for a job."""
    filepath = os.path.join(APPLICATIONS_DIR, job_id, "tailored_cv.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _build_plain_text_cv(cv_data):
    """Convert structured CV JSON into a clean plain-text resume string."""
    lines = []

    name = cv_data.get("full_name", "")
    if name:
        lines.append(name)

    contact_parts = []
    if cv_data.get("email"):
        contact_parts.append(cv_data["email"])
    if cv_data.get("phone"):
        contact_parts.append(cv_data["phone"])
    if cv_data.get("location"):
        contact_parts.append(cv_data["location"])
    if contact_parts:
        lines.append(" | ".join(contact_parts))

    links = cv_data.get("links", [])
    if isinstance(links, dict):
        links = [str(v) for v in links.values() if v]
    if isinstance(links, list) and links:
        lines.append(" | ".join(links))

    lines.append("")

    if cv_data.get("summary"):
        lines.append("PROFESSIONAL SUMMARY")
        lines.append(cv_data["summary"])
        lines.append("")

    skills = cv_data.get("skills", [])
    if isinstance(skills, list) and skills:
        lines.append("TECHNICAL SKILLS")
        lines.append(", ".join(skills))
        lines.append("")

    experience = cv_data.get("experience", [])
    if isinstance(experience, list) and experience:
        lines.append("EXPERIENCE")
        for exp in experience:
            role = exp.get("role") or exp.get("title") or exp.get("job_title") or exp.get("position", "")
            company = exp.get("company") or exp.get("organization", "")
            duration = exp.get("duration") or exp.get("dates") or exp.get("period", "")
            lines.append(f"{role} at {company} ({duration})")

            desc = exp.get("description") or exp.get("bullet_points") or exp.get("highlights")
            if isinstance(desc, list):
                for bullet in desc:
                    lines.append(f"  - {bullet}")
            elif isinstance(desc, str):
                for bullet in desc.split("\n"):
                    b = bullet.strip().lstrip("-•* ").strip()
                    if b:
                        lines.append(f"  - {b}")
            lines.append("")

    education = cv_data.get("education", [])
    if isinstance(education, list) and education:
        lines.append("EDUCATION")
        for edu in education:
            degree = edu.get("degree") or edu.get("title") or edu.get("program", "")
            institution = edu.get("institution") or edu.get("school") or edu.get("university", "")
            year = edu.get("year") or edu.get("duration") or edu.get("dates", "")
            lines.append(f"{degree} - {institution} ({year})")
        lines.append("")

    projects = cv_data.get("projects", [])
    if isinstance(projects, list) and projects:
        lines.append("PROJECTS")
        for proj in projects:
            title = proj.get("title") or proj.get("name", "")
            lines.append(title)
            desc = proj.get("description")
            if isinstance(desc, list):
                for bullet in desc:
                    lines.append(f"  - {bullet}")
            elif isinstance(desc, str):
                for bullet in desc.split("\n"):
                    b = bullet.strip().lstrip("-•* ").strip()
                    if b:
                        lines.append(f"  - {b}")
            lines.append("")

    certifications = cv_data.get("certifications", [])
    if isinstance(certifications, list) and certifications:
        lines.append("CERTIFICATIONS")
        for cert in certifications:
            lines.append(f"  - {cert}")
        lines.append("")

    return "\n".join(lines)


def _generate_cv_text_file(cv_data, job_id):
    """Generate a plain-text .txt file of the tailored CV for form file uploads."""
    text = _build_plain_text_cv(cv_data)
    temp_dir = os.path.join(APPLICATIONS_DIR, job_id)
    os.makedirs(temp_dir, exist_ok=True)
    filepath = os.path.join(temp_dir, "tailored_cv.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return filepath


def _generate_cover_letter_docx_file(cover_letter, job_id):
    """
    Generate a proper .docx cover letter document for form uploads — not every ATS
    accepts a plain .txt file, so a real Word document is the safer default for
    the cover-letter file field.
    """
    from docx import Document

    temp_dir = os.path.join(APPLICATIONS_DIR, job_id)
    os.makedirs(temp_dir, exist_ok=True)
    filepath = os.path.join(temp_dir, "cover_letter.docx")

    doc = Document()
    for paragraph in (cover_letter or "").split("\n"):
        doc.add_paragraph(paragraph)
    doc.save(filepath)
    return filepath


def _get_latest_uploaded_resume():
    """Return the absolute path to the latest uploaded PDF or DOCX resume in UPLOADS_DIR."""
    if not os.path.exists(UPLOADS_DIR):
        return None
    files = []
    for ext in ("*.pdf", "*.docx", "*.doc"):
        files.extend(glob.glob(os.path.join(UPLOADS_DIR, ext)))
    if not files:
        # Check any files
        files = [os.path.join(UPLOADS_DIR, f) for f in os.listdir(UPLOADS_DIR) if os.path.isfile(os.path.join(UPLOADS_DIR, f))]
    if not files:
        return None
    # Sort by modification time desc
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


class SequentialApplier:
    # Recycling the browser periodically during a long bulk-apply run reduces
    # the accumulated resource/state pressure (leftover renderer processes,
    # long-lived profile state) that likely contributes to the kind of full
    # ChromeDriver-connection-lost crash confirmed live on a Tata Consultancy
    # Services job — a fresh, clean session every so often is cheap insurance
    # against that class of failure recurring.
    RESTART_EVERY_N_JOBS = 15

    def __init__(self, cv_data, cover_letter_func):
        self.cv_data = cv_data
        self.cover_letter_func = cover_letter_func
        self.driver = None
        self.current_job_id = None
        self._jobs_since_restart = 0

    def init_browser(self):
        if not self.driver:
            options = webdriver.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--disable-popup-blocking')
            options.add_experimental_option('excludeSwitches', ['enable-automation'])
            options.add_experimental_option('useAutomationExtension', False)

            # Onboarding & first-run suppression to bypass welcome/What's New screens
            options.add_argument('--no-first-run')
            options.add_argument('--no-default-browser-check')
            options.add_argument('--disable-features=ChromeWhatsNewUI')
            options.add_argument('--disable-whats-new-ui')
            options.add_argument('--disable-welcome-ui')

            # Using persistent user data directory (same profile undetected-chromedriver used,
            # so existing LinkedIn/Indeed logins carry over)
            user_data_dir = os.path.join(os.getcwd(), "user_data_uc")
            os.makedirs(user_data_dir, exist_ok=True)
            options.add_argument(f"--user-data-dir={user_data_dir}")

            # Selenium Manager (built into Selenium 4.6+) auto-resolves a chromedriver
            # build matching whatever Chrome version is actually installed.
            self.driver = webdriver.Chrome(options=options)

            # Stealth overriding
            self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                    Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                """
            })

    def _try_fill(self, selector, value, skip_if_filled=True):
        """
        Try to fill a form field matching the selector, returns True if found.
        Skips elements that already hold a value by default — without this, a later,
        broader selector (e.g. the address fallback matching an "Email address"
        placeholder) can silently overwrite a field an earlier, more specific
        selector already filled correctly.
        """
        try:
            by_type = By.XPATH if selector.startswith("/") or selector.startswith("xpath:") else By.CSS_SELECTOR
            actual_sel = selector[6:] if selector.startswith("xpath:") else selector
            elements = self.driver.find_elements(by_type, actual_sel)
            for el in elements:
                if el.is_displayed():
                    if skip_if_filled and (el.get_attribute("value") or "").strip():
                        continue
                    self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                    time.sleep(0.15)
                    try:
                        el.click()
                    except:
                        pass
                    # Force-clear via the framework-aware native setter first. Keyboard-only
                    # clearing (Ctrl+A/Backspace) can silently fail on some custom/masked
                    # React inputs, leaving old content in place for new keystrokes to land
                    # after instead of replacing.
                    try:
                        self.driver.execute_script(_NATIVE_SET_VALUE_JS, el, "")
                    except Exception:
                        pass
                    try:
                        el.send_keys(Keys.CONTROL + "a")
                        el.send_keys(Keys.BACKSPACE)
                    except Exception:
                        try:
                            el.clear()
                        except Exception:
                            pass
                    time.sleep(0.1)
                    el.send_keys(str(value))
                    # Verify the field actually holds exactly what was typed — if a
                    # framework re-render restored stale content mid-type, force it directly.
                    final_value = (el.get_attribute("value") or "").strip()
                    if final_value != str(value).strip():
                        try:
                            self.driver.execute_script(_NATIVE_SET_VALUE_JS, el, str(value))
                        except Exception:
                            pass
                    print(f"  [OK] Filled field: '{selector}' -> '{value}'")

                    if hasattr(self, "current_job_id") and self.current_job_id:

                        log_step(self.current_job_id, "Generic Filler", f"  [OK] Filled field: '{selector}' -> '{value}'")
                    return True
        except Exception as e:
            print(f"  [DEBUG] Error filling selector '{selector}': {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", f"  [DEBUG] Error filling selector '{selector}': {e}")
        return False

    def _try_fill_any(self, selectors, value):
        """Try multiple selectors in order, fill the first match."""
        for selector in selectors:
            if self._try_fill(selector, value):
                return True
        return False

    def _mark_uploaded(self, el):
        """
        Best-effort — some ATS widgets (e.g. Dice's cover-letter dropzone) swap the
        <input> out of the DOM the instant a file is set (React re-renders the dropzone
        into a "file attached" card), which makes this throw StaleElementReferenceException
        even though the upload itself already succeeded. Never let that turn a real
        success into a reported failure — it's purely a nice-to-have to prevent re-use.
        """
        try:
            self.driver.execute_script("arguments[0].setAttribute('data-bot-uploaded', '1');", el)
        except Exception:
            pass

    def _nearby_container_text(self, el):
        """
        Text of the nearest reasonably-scoped ancestor container, lowercased — used to
        tell an unlabeled file input's actual purpose (Resume vs Cover letter) from
        surrounding heading text when the input has no name/id/aria-label at all (e.g.
        Dice's cover-letter dropzone input has none of those, only a "Cover letter"
        heading a few DOM levels up).
        """
        try:
            return (self.driver.execute_script(
                """
                let node = arguments[0];
                for (let i = 0; i < 5 && node; i++) {
                    const text = (node.innerText || node.textContent || '').trim();
                    if (text && text.length < 300) return text;
                    node = node.parentElement;
                }
                return '';
                """,
                el,
            ) or "").lower()
        except Exception:
            return ""

    def _try_upload_file(self, filepath, kind="resume"):
        """
        Try to find a file input and upload a file into it. `kind` is "resume" or
        "cover_letter" — used to prefer the matching input when a form has more than
        one file field, and each input is marked after use so resume/cover-letter
        uploads never collide on the same field.
        """
        try:
            if kind == "cover_letter":
                preferred_selectors = [
                    "input[type='file'][name*='cover' i]",
                    "input[type='file'][id*='cover' i]",
                    "input[type='file'][aria-label*='cover' i]",
                ]
            else:
                preferred_selectors = [
                    "input[type='file'][name*='cv' i]",
                    "input[type='file'][name*='resume' i]",
                    "input[type='file'][id*='cv' i]",
                    "input[type='file'][id*='resume' i]",
                    "input[type='file'][aria-label*='resume' i]",
                    "input[type='file'][aria-label*='cv' i]",
                ]

            def _unfilled(elements):
                return [el for el in elements if not el.get_attribute("data-bot-uploaded")]

            for selector in preferred_selectors:
                elements = _unfilled(self.driver.find_elements(By.CSS_SELECTOR, selector))
                if elements:
                    elements[0].send_keys(os.path.abspath(filepath))
                    print(f"  [OK] Uploaded {kind} file via: {selector}")

                    if hasattr(self, "current_job_id") and self.current_job_id:

                        log_step(self.current_job_id, "Generic Filler", f"  [OK] Uploaded {kind} file via: {selector}")
                    self._mark_uploaded(elements[0])
                    time.sleep(1.5)  # let the dropzone's JS process/validate the upload before moving on
                    return True

            # No name/id/aria-label match — some ATS forms render a completely unlabeled
            # <input type="file"> (Dice's cover-letter dropzone, for one), only
            # distinguishable by the heading text of its surrounding container. Check that
            # BEFORE any accept-type guess below — otherwise a resume upload can accidentally
            # grab a cover-letter-only input just because both happen to accept .pdf/.doc,
            # silently attaching the resume as the "cover letter" and leaving the real one
            # unsent (this exact bug was confirmed live against Dice's apply wizard).
            own_keyword = "cover" if kind == "cover_letter" else "resume"
            other_keyword = "resume" if kind == "cover_letter" else "cover"
            for el in _unfilled(self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")):
                context = self._nearby_container_text(el)
                if own_keyword in context and other_keyword not in context:
                    el.send_keys(os.path.abspath(filepath))
                    print(f"  [OK] Uploaded {kind} file via context-matched input (nearby text: '{own_keyword}')")

                    if hasattr(self, "current_job_id") and self.current_job_id:

                        log_step(self.current_job_id, "Generic Filler", f"  [OK] Uploaded {kind} file via context-matched input (nearby text: '{own_keyword}')")
                    self._mark_uploaded(el)
                    time.sleep(1.5)
                    return True

            # Accept-type guess for resume only, still excluding anything whose surrounding
            # text clearly belongs to the cover letter (see reasoning above).
            if kind != "cover_letter":
                for selector in ["input[type='file'][accept*='pdf' i]", "input[type='file'][accept*='doc' i]"]:
                    for el in _unfilled(self.driver.find_elements(By.CSS_SELECTOR, selector)):
                        if "cover" in self._nearby_container_text(el):
                            continue
                        el.send_keys(os.path.abspath(filepath))
                        print(f"  [OK] Uploaded {kind} file via: {selector}")

                        if hasattr(self, "current_job_id") and self.current_job_id:

                            log_step(self.current_job_id, "Generic Filler", f"  [OK] Uploaded {kind} file via: {selector}")
                        self._mark_uploaded(el)
                        time.sleep(1.5)
                        return True

            # Last resort: a plain, single, generic file input with nothing else to go on.
            # Resume fields consistently render before cover-letter fields in DOM order, so
            # calling this for "resume" first and "cover_letter" second naturally targets
            # the right one either way — except when the only candidate is clearly the
            # cover letter's own input (the Dice case), which a resume upload must never take.
            all_file_inputs = _unfilled(self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']"))
            if kind != "cover_letter":
                all_file_inputs = [el for el in all_file_inputs if "cover" not in self._nearby_container_text(el)]
            if all_file_inputs:
                all_file_inputs[0].send_keys(os.path.abspath(filepath))
                print(f"  [OK] Uploaded {kind} file via first unfilled file input")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"  [OK] Uploaded {kind} file via first unfilled file input")
                self._mark_uploaded(all_file_inputs[0])
                time.sleep(1.5)
                return True

        except Exception as e:
            print(f"  [WARN] File upload failed ({kind}): {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", f"  [WARN] File upload failed ({kind}): {e}")
        return False

    def _recover_dead_window(self):
        """
        True if the driver was successfully switched to a live tab after its
        current one was closed out from under it. Confirmed live on a
        Randstad Digital job (via Dice) and mirrored here for Indeed's
        external-redirect path (IndeedExternalApplier delegates to this
        class): an intermediate redirector tab can open the REAL application
        in a further new tab and then close itself a moment later — every
        operation on the now-dead handle then fails with "no such window:
        target window already closed", including the apply-button search and
        every single field-fill selector, which cascaded into what looked
        like the browser just closing mid-application. Returns False if the
        current tab is still fine (nothing to recover) or if no tabs remain.
        """
        try:
            self.driver.current_window_handle
            return False
        except Exception:
            pass
        try:
            remaining = self.driver.window_handles
        except Exception:
            remaining = []
        if not remaining:
            return False
        try:
            self.driver.switch_to.window(remaining[-1])
            print("  [OK] Previous tab closed itself — switched to the new active tab.")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", "  [OK] Previous tab closed itself — switched to the new active tab.")
            return True
        except Exception:
            return False

    def _click_initial_apply_button(self):
        """Locate and click the initial 'Apply' / 'Apply Now' button on an external site if the form is not visible yet."""
        try:
            # First, check if form elements are already visible. If yes, we don't need to click any button!
            form_elements = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "input[name*='name' i], input[id*='name' i], input[type='email'], input[type='file'], textarea"
            )
            if any(el.is_displayed() for el in form_elements):
                print("  Form fields already visible on landing. Skipping initial apply click.")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", "  Form fields already visible on landing. Skipping initial apply click.")
                return True
        except Exception:
            pass

        print("  Form fields not visible. Searching for initial 'Apply' / 'Apply Now' button on portal...")


        if hasattr(self, "current_job_id") and self.current_job_id:


            log_step(self.current_job_id, "Generic Filler", "  Form fields not visible. Searching for initial 'Apply' / 'Apply Now' button on portal...")

        # No "not Easy Apply" exclusion here — this only ever runs on a site's OWN
        # page for sources without a dedicated agent (e.g. Dice), or on the external
        # redirect target for LinkedIn/Indeed (never their own page, which their
        # dedicated agents already handle) — so an "Easy Apply" button here (Dice
        # uses that exact label) is always the right thing to click, not LinkedIn's.
        #
        # The find-and-click happens as a single atomic JS call rather than
        # Selenium find_elements() + a separate .click() round-trip: some job
        # pages (e.g. Dice) keep re-rendering their DOM continuously for several
        # seconds after load (ads/analytics/hydration), which turns a
        # Selenium WebElement handle stale in the gap between finding it and
        # clicking it — including the JS-click-by-reference fallback, since that
        # still reuses the same now-detached handle. Re-querying by text from
        # inside one script call sidesteps that entirely.
        find_and_click_js = """
            function isVisible(el) {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
            }
            const candidates = Array.from(document.querySelectorAll('a, button'));

            function tryClick(matcher) {
                for (const el of candidates) {
                    const text = (el.innerText || el.textContent || '').trim();
                    const aria = (el.getAttribute('aria-label') || '').trim();
                    const href = el.getAttribute('href') || '';
                    const haystack = (text + ' ' + aria + ' ' + href).toLowerCase();
                    if (!matcher(haystack)) continue;
                    if (el.disabled) continue;
                    if (!isVisible(el)) continue;
                    el.scrollIntoView({block: 'center'});
                    el.click();
                    return {clicked: true, text: text, href: href};
                }
                return null;
            }

            // Priority 1: anything mentioning "apply" — unchanged from before, this is
            // exactly what matches Dice's "Easy Apply" button and every other site's
            // "Apply"/"Apply Now", so it keeps matching first, same as always.
            let result = tryClick(h => h.includes('apply'));
            if (!result) {
                // Priority 2 (fallback only): some external ATS forms use different
                // first-screen wording instead of "apply" at all.
                result = tryClick(h => h.includes('continue') || h.includes('get started') || h.includes('start application'));
            }
            return result || {clicked: false};
        """

        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                result = self.driver.execute_script(find_and_click_js) or {}
            except Exception as e:
                error_str = str(e).lower()
                if "no such window" in error_str or "target window already closed" in error_str:
                    if self._recover_dead_window():
                        time.sleep(1.0)
                        continue
                    break
                print(f"  [DEBUG] Apply-button search script error: {e}")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"  [DEBUG] Apply-button search script error: {e}")
                result = {}

            if result.get("clicked"):
                print(f"  [OK] Clicked initial apply button: '{result.get('text', '')}' (href='{result.get('href', '')}')")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"  [OK] Clicked initial apply button: '{result.get('text', '')}' (href='{result.get('href', '')}')")
                time.sleep(3.5)  # Wait for page or form container to render
                self.wait_for_cloudflare()
                return True

            time.sleep(0.4)

        print("  [WARN] No initial apply button found or clicked.")


        if hasattr(self, "current_job_id") and self.current_job_id:


            log_step(self.current_job_id, "Generic Filler", "  [WARN] No initial apply button found or clicked.")
        return False

    def _select_dropdown_by_text_variants(self, select_el, variants):
        """Select the first <select> option whose visible text matches (exact, then substring) any variant."""
        try:
            sel_obj = Select(select_el)
            options = sel_obj.options
            lowered = [v.strip().lower() for v in variants if v]
            for opt in options:
                if opt.text.strip().lower() in lowered:
                    sel_obj.select_by_visible_text(opt.text)
                    return True
            for opt in options:
                opt_l = opt.text.strip().lower()
                if any(v and v in opt_l for v in lowered):
                    sel_obj.select_by_visible_text(opt.text)
                    return True
        except Exception:
            pass
        return False

    def _fill_identity_and_links(self, tailored_cv):
        """Deterministic pass: name, email, phone, address/location, state/country, and profile links.
        Returns how many distinct fields were successfully filled/selected, so callers can detect a
        step where the page didn't match any known selector at all (worth a loud warning, not a
        silent no-op) instead of just a partially-filled one."""
        full_name = tailored_cv.get("full_name", "")
        email = tailored_cv.get("email", "") or CANDIDATE_EMAIL
        phone = tailored_cv.get("phone", "")
        location = tailored_cv.get("location", "")
        summary = tailored_cv.get("summary", "")

        name_parts = full_name.split() if full_name else []
        first_name = name_parts[0] if name_parts else ""
        last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""

        # Confirmed live on a Snowflake Data Engineer job and an Amtrak job
        # (via LinkedIn's external-redirect path; the same tab-lifecycle
        # pattern applies here for Indeed via IndeedExternalApplier): if the
        # current tab closed itself out from under us (an intermediate
        # redirector tab that opens the real form in a further new tab, then
        # closes), every one of the dozens of selector attempts below would
        # fail identically with "no such window: target window already
        # closed" — a pure log-spam cascade that looked like the whole
        # browser just died. Check once up front instead: recover to a live
        # tab if possible, or skip this entire fill pass cleanly if nothing
        # is left to recover to.
        try:
            self.driver.current_window_handle
        except Exception:
            if not self._recover_dead_window():
                return 0

        filled_count = 0

        if self._try_fill_any([
            "input[name*='full_name' i]", "input[id*='full_name' i]",
            "input[name*='fullname' i]", "input[id*='fullname' i]",
            "input[name*='display_name' i]", "input[id*='display_name' i]",
            "input[name*='name' i][name*='full' i]",
            "input[placeholder*='full name' i]",
        ], full_name):
            filled_count += 1

        if self._try_fill_any([
            "input[name*='first_name' i]", "input[id*='first_name' i]",
            "input[name*='firstname' i]", "input[id*='firstname' i]",
            "input[name*='given' i]", "input[id*='given' i]",
            "input[placeholder*='first name' i]",
        ], first_name):
            filled_count += 1

        if self._try_fill_any([
            "input[name*='last_name' i]", "input[id*='last_name' i]",
            "input[name*='lastname' i]", "input[id*='lastname' i]",
            "input[name*='surname' i]", "input[id*='surname' i]",
            "input[name*='family' i]", "input[id*='family' i]",
            "input[placeholder*='last name' i]",
        ], last_name):
            filled_count += 1

        if self._try_fill_any([
            "input[type='email']",
            "input[name*='email' i]", "input[id*='email' i]",
            "input[placeholder*='email' i]",
        ], email):
            filled_count += 1

        # Confirm/verify-email fields (e.g. Kforce's "Verify Email") never appear in CV
        # data, so they must be deterministically mirrored from the primary email rather
        # than left for the LLM pass, which has no visibility into what was already typed
        # into a different field on the same page.
        if email:
            verify_filled = self._try_fill_any([
                "input[name*='email' i][name*='verify' i]", "input[id*='email' i][id*='verify' i]",
                "input[name*='email' i][name*='confirm' i]", "input[id*='email' i][id*='confirm' i]",
                "input[name*='confirm' i][name*='email' i]", "input[id*='confirm' i][id*='email' i]",
                "input[placeholder*='verify email' i]", "input[placeholder*='confirm email' i]",
                "input[placeholder*='re-enter email' i]", "input[placeholder*='reenter email' i]",
            ], email)
            if not verify_filled:
                # Fall back to any other still-empty type=email input — covers a second/confirm
                # email field that doesn't literally say "verify"/"confirm" in its name.
                verify_filled = self._try_fill_any(["input[type='email']"], email)
            if verify_filled:
                filled_count += 1

        if phone:
            if self._try_fill_any([
                "input[type='tel']",
                "input[name*='phone' i]", "input[id*='phone' i]",
                "input[name*='mobile' i]", "input[id*='mobile' i]",
                "input[name*='telephone' i]", "input[id*='telephone' i]",
                "input[placeholder*='phone' i]", "input[placeholder*='mobile' i]",
            ], phone):
                filled_count += 1

        if location:
            # Covers plain "location/city" fields as well as street-address-style
            # fields (Workday/Greenhouse/Lensa commonly ask "Address"/"Address Line 1").
            # Explicitly excludes anything mentioning "email" — a placeholder like
            # "Email address" contains the substring "address" and would otherwise
            # match these selectors and get overwritten with the location value.
            if self._try_fill_any([
                "input[name*='location' i]:not([name*='email' i])",
                "input[id*='location' i]:not([id*='email' i])",
                "input[name*='city' i]:not([name*='email' i])",
                "input[id*='city' i]:not([id*='email' i])",
                "input[name*='address' i]:not([name*='email' i])",
                "input[id*='address' i]:not([id*='email' i])",
                "input[name*='street' i]:not([name*='email' i])",
                "input[id*='street' i]:not([id*='email' i])",
                "input[placeholder*='city' i]:not([placeholder*='email' i])",
                "input[placeholder*='address' i]:not([placeholder*='email' i]):not([type='email'])",
                "input[placeholder*='location' i]:not([placeholder*='email' i])",
            ], location):
                filled_count += 1

            # State/Country <select> dropdowns — very common on US ATS forms (Kforce,
            # Workday, etc.) and never handled by the plain text-field pass above since
            # a <select> has no "value" to type into. Parsed from "City, ST" (or
            # "City, State") in the CV's location string.
            state_token = location.split(",")[-1].strip() if "," in location else ""
            if state_token:
                state_name = US_STATE_ABBR_TO_NAME.get(state_token.upper(), "")
                variants = [state_token, state_name] if state_name else [state_token]
                for sel_el in self.driver.find_elements(
                    By.CSS_SELECTOR,
                    "select[name*='state' i]:not([name*='country' i]), select[id*='state' i]:not([id*='country' i])"
                ):
                    if sel_el.is_displayed():
                        if self._select_dropdown_by_text_variants(sel_el, variants):
                            print(f"  [OK] State dropdown selected: {state_token}")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Generic Filler", f"  [OK] State dropdown selected: {state_token}")
                            filled_count += 1
                        break

            # Country — only touched if it's still on a placeholder ("Select...", etc);
            # many forms already default to a sensible value and shouldn't be clobbered.
            for sel_el in self.driver.find_elements(By.CSS_SELECTOR, "select[name*='country' i], select[id*='country' i]"):
                if sel_el.is_displayed():
                    try:
                        current = Select(sel_el).first_selected_option.text.strip()
                    except Exception:
                        current = ""
                    if not current or current.lower() in ("select...", "select one", "please select", "-- select --", ""):
                        if self._select_dropdown_by_text_variants(
                            sel_el, ["United States", "United States of America", "USA", "US"]
                        ):
                            print("  [OK] Country dropdown selected: United States")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Generic Filler", "  [OK] Country dropdown selected: United States")
                            filled_count += 1
                    break

        links = tailored_cv.get('links', [])
        if isinstance(links, dict):
            links = [str(v) for v in links.values() if v]
        if isinstance(links, list):
            for link in links:
                l_lower = link.lower()
                if 'linkedin' in l_lower:
                    if self._try_fill_any([
                        "input[name*='linkedin' i]", "input[id*='linkedin' i]",
                        "input[placeholder*='linkedin' i]",
                        "input[name*='link' i][name*='in' i]",
                    ], link):
                        filled_count += 1
                elif 'github' in l_lower:
                    if self._try_fill_any([
                        "input[name*='github' i]", "input[id*='github' i]",
                        "input[placeholder*='github' i]",
                        "input[name*='git' i]",
                    ], link):
                        filled_count += 1
                elif 'portfolio' in l_lower or 'website' in l_lower:
                    if self._try_fill_any([
                        "input[name*='portfolio' i]", "input[id*='portfolio' i]",
                        "input[name*='website' i]", "input[id*='website' i]",
                        "input[placeholder*='website' i]",
                        "input[placeholder*='portfolio' i]",
                        "input[name*='personal' i][name*='site' i]",
                    ], link):
                        filled_count += 1

        if summary:
            if self._try_fill_any([
                "textarea[name*='summary' i]", "textarea[id*='summary' i]",
                "textarea[name*='about' i]", "textarea[id*='about' i]",
                "textarea[placeholder*='summary' i]",
            ], summary):
                filled_count += 1

        return filled_count

    def _click_consent_checkboxes(self):
        try:
            consent_selectors = [
                "input[type='checkbox'][name*='privacy' i]",
                "input[type='checkbox'][name*='consent' i]",
                "input[type='checkbox'][name*='terms' i]",
                "input[type='checkbox'][id*='privacy' i]",
                "input[type='checkbox'][id*='consent' i]",
                "input[type='checkbox'][id*='agree' i]",
                "input[type='checkbox'][value*='agree' i]",
            ]
            for sel in consent_selectors:
                for el in self.driver.find_elements(By.CSS_SELECTOR, sel):
                    if el.is_displayed() and not el.is_selected():
                        self.driver.execute_script("arguments[0].click();", el)
                        print(f"  [OK] Clicked consent checkbox: {sel}")

                        if hasattr(self, "current_job_id") and self.current_job_id:

                            log_step(self.current_job_id, "Generic Filler", f"  [OK] Clicked consent checkbox: {sel}")
        except Exception as ce:
            print(f"  [WARN] Checkbox clicking error: {ce}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", f"  [WARN] Checkbox clicking error: {ce}")

    def _fill_known_dropdowns(self):
        """Rule-based auto-fill for the two dropdown questions we can always answer safely."""
        try:
            select_elements = self.driver.find_elements(By.CSS_SELECTOR, "select")
            for sel in select_elements:
                if not sel.is_displayed():
                    continue
                aria = (sel.get_attribute("aria-label") or "").lower()
                name = (sel.get_attribute("name") or "").lower()
                text_content = (sel.text or "").lower()

                is_sponsorship = any(k in aria or k in name or k in text_content for k in ("sponsor", "visa", "sponsorship"))
                is_auth = any(k in aria or k in name or k in text_content for k in ("authorized", "legally", "permit"))
                is_relocate = "relocat" in aria or "relocat" in name or "relocat" in text_content
                is_gender = (
                    "gender" in aria or "gender" in name or "gender" in text_content
                    or "sex" in aria.split() or "sex" in name.split() or "sex" in text_content.split()
                )

                sel_obj = Select(sel)

                if is_sponsorship:
                    for opt in sel_obj.options:
                        opt_text = (opt.text or "").lower()
                        if opt_text.strip() in ("no", "false", "n"):
                            sel_obj.select_by_visible_text(opt.text)
                            print(f"  [OK] Sponsorship dropdown auto-filled: {opt.text}")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Generic Filler", f"  [OK] Sponsorship dropdown auto-filled: {opt.text}")
                            break
                elif is_auth:
                    for opt in sel_obj.options:
                        opt_text = (opt.text or "").lower()
                        if opt_text.strip() in ("yes", "true", "y"):
                            sel_obj.select_by_visible_text(opt.text)
                            print(f"  [OK] Work Auth dropdown auto-filled: {opt.text}")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Generic Filler", f"  [OK] Work Auth dropdown auto-filled: {opt.text}")
                            break
                elif is_relocate:
                    # Fixed personal-preference answer (not derivable from a CV) — set once,
                    # applied consistently across every application.
                    for opt in sel_obj.options:
                        opt_text = (opt.text or "").lower()
                        if opt_text.strip() in ("no", "false", "n"):
                            sel_obj.select_by_visible_text(opt.text)
                            print(f"  [OK] Relocation dropdown auto-filled: {opt.text}")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Generic Filler", f"  [OK] Relocation dropdown auto-filled: {opt.text}")
                            break
                elif is_gender:
                    # Never guess a gender — only act if a "decline to answer" style
                    # option actually exists; otherwise leave it for the LLM/human rather
                    # than pick Male/Female with no CV basis for either.
                    for opt in sel_obj.options:
                        opt_text = (opt.text or "").lower()
                        if any(k in opt_text for k in DECLINE_TO_ANSWER_KEYWORDS):
                            sel_obj.select_by_visible_text(opt.text)
                            print(f"  [OK] Gender/self-ID dropdown auto-filled: {opt.text}")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Generic Filler", f"  [OK] Gender/self-ID dropdown auto-filled: {opt.text}")
                            break
        except Exception as se:
            print(f"  [WARN] Dropdown auto-filling error: {se}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", f"  [WARN] Dropdown auto-filling error: {se}")

    def _radio_label(self, el):
        gid = el.get_attribute("id") or ""
        label = ""
        if gid:
            try:
                label = self.driver.find_element(By.CSS_SELECTOR, f"label[for='{gid}']").text.strip()
            except Exception:
                pass
        return (label or el.get_attribute("value") or "").strip().lower()

    def _fill_known_radio_groups(self):
        """
        Rule-based auto-fill for screening questions whose answer is a fixed personal
        preference or a legally-optional non-answer rather than a CV fact (e.g. "willing
        to relocate?", gender/voluntary self-identification) — a CV has no way to answer
        these, so without this they're left for the LLM pass, which has no real basis to
        answer them either and correctly (per its own rules) leaves them blank rather
        than guess.
        """
        try:
            seen_names = set()
            for r in self.driver.find_elements(By.CSS_SELECTOR, "input[type='radio']"):
                name = r.get_attribute("name") or ""
                if not name or name in seen_names:
                    continue
                group = self.driver.find_elements(By.CSS_SELECTOR, f"input[type='radio'][name='{name}']")
                if not any(g.is_displayed() for g in group):
                    continue
                seen_names.add(name)
                if any(g.is_selected() for g in group):
                    continue

                context = self._nearby_container_text(group[0])

                if "relocat" in context:
                    for g in group:
                        if self._radio_label(g) == "no" and not g.is_selected():
                            self.driver.execute_script("arguments[0].click();", g)
                            print(f"  [OK] Relocation radio auto-filled ('{name}'): No")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Generic Filler", f"  [OK] Relocation radio auto-filled ('{name}'): No")
                            break
                elif "gender" in context or "sex" in context.split():
                    # Never guess a gender — only act if a "decline to answer" style
                    # option actually exists; otherwise leave it for the LLM/human rather
                    # than pick Male/Female with no CV basis for either.
                    for g in group:
                        label = self._radio_label(g)
                        if any(k in label for k in DECLINE_TO_ANSWER_KEYWORDS) and not g.is_selected():
                            self.driver.execute_script("arguments[0].click();", g)
                            print(f"  [OK] Gender/self-ID radio auto-filled ('{name}'): {label}")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Generic Filler", f"  [OK] Gender/self-ID radio auto-filled ('{name}'): {label}")
                            break
        except Exception as e:
            print(f"  [WARN] Known radio-group auto-fill error: {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", f"  [WARN] Known radio-group auto-fill error: {e}")

    def _fill_date_inputs(self):
        """
        Deterministically fill empty <input type="date"> fields with today's date —
        native date pickers ignore plain send_keys text entirely, so without this
        they're silently skipped by every other fill pass (all of which only know how
        to type into text-like inputs).
        """
        try:
            today_iso = time.strftime("%Y-%m-%d")
            for el in self.driver.find_elements(By.CSS_SELECTOR, "input[type='date']"):
                if not el.is_displayed():
                    continue
                if (el.get_attribute("value") or "").strip():
                    continue
                self.driver.execute_script(_NATIVE_SET_VALUE_JS, el, today_iso)
                print(f"  [OK] Date field filled with today's date: {today_iso}")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"  [OK] Date field filled with today's date: {today_iso}")
        except Exception as e:
            print(f"  [WARN] Date field auto-fill error: {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", f"  [WARN] Date field auto-fill error: {e}")

    def _fill_range_sliders(self, default_percent="0"):
        """
        Deterministically set <input type="range"> sliders (e.g. '% of days willing to
        travel') to a fixed default — sliders have no text value to type into, so
        send_keys never works on them; the value must be set directly via JS.
        """
        try:
            for el in self.driver.find_elements(By.CSS_SELECTOR, "input[type='range']"):
                if not el.is_displayed():
                    continue
                if (el.get_attribute("value") or "").strip() == str(default_percent):
                    continue
                self.driver.execute_script(_NATIVE_SET_VALUE_JS, el, str(default_percent))
                print(f"  [OK] Range slider set to: {default_percent}")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"  [OK] Range slider set to: {default_percent}")
        except Exception as e:
            print(f"  [WARN] Range slider auto-fill error: {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", f"  [WARN] Range slider auto-fill error: {e}")

    def _field_label(self, el):
        try:
            eid = el.get_attribute("id") or ""
            if eid:
                try:
                    lbl = self.driver.find_element(By.CSS_SELECTOR, f"label[for='{eid}']")
                    if lbl.text.strip():
                        return lbl.text.strip()
                except Exception:
                    pass
            try:
                anc = el.find_element(By.XPATH, "ancestor::label")
                if anc.text.strip():
                    return anc.text.strip()
            except Exception:
                pass
            return el.get_attribute("aria-label") or el.get_attribute("placeholder") or ""
        except Exception:
            return ""

    def _collect_unfilled_fields(self):
        """
        Scan the whole page for visible fields still needing a value/selection —
        including arbitrary <select> dropdowns, radio groups, and checkboxes beyond
        the deterministic identity/consent/sponsorship ones already handled — so the
        LLM pass below can answer them intelligently from the CV.
        """
        fields = []
        seen_radio_groups = set()
        try:
            elements = self.driver.find_elements(
                By.CSS_SELECTOR,
                "input:not([type='hidden']):not([type='file']), select, textarea"
            )
            for el in elements:
                try:
                    if not el.is_displayed():
                        continue
                    tag = el.tag_name.lower()
                    itype = (el.get_attribute("type") or tag).lower()
                    name = el.get_attribute("name") or ""
                    eid = el.get_attribute("id") or ""

                    if itype == "radio":
                        group_key = name or eid
                        if not group_key or group_key in seen_radio_groups:
                            continue
                        seen_radio_groups.add(group_key)
                        group_radios = (
                            self.driver.find_elements(By.CSS_SELECTOR, f"input[type='radio'][name='{name}']")
                            if name else [el]
                        )
                        if any(r.is_selected() for r in group_radios):
                            continue
                        options = []
                        for r in group_radios:
                            rid = r.get_attribute("id") or ""
                            label = ""
                            if rid:
                                try:
                                    label = self.driver.find_element(By.CSS_SELECTOR, f"label[for='{rid}']").text.strip()
                                except Exception:
                                    pass
                            label = label or r.get_attribute("value") or ""
                            if label:
                                options.append(label)
                        if options:
                            fields.append({"kind": "radio", "label": self._field_label(el), "name": name, "id": "", "options": options})
                        continue

                    if itype == "checkbox":
                        if el.is_selected():
                            continue
                        fields.append({"kind": "checkbox", "label": self._field_label(el), "name": name, "id": eid, "options": ["check", "leave unchecked"]})
                        continue

                    if tag == "select":
                        try:
                            selected_text = Select(el).first_selected_option.text.strip()
                        except Exception:
                            selected_text = ""
                        if selected_text and selected_text.lower() not in ("select...", "select one", "please select", "-- select --", ""):
                            continue
                        try:
                            options = [o.text.strip() for o in Select(el).options if o.text.strip()]
                        except Exception:
                            options = []
                        if not options:
                            continue
                        fields.append({"kind": "select", "label": self._field_label(el), "name": name, "id": eid, "options": options})
                        continue

                    current = (el.get_attribute("value") or "").strip()
                    if current:
                        continue
                    label_text = self._field_label(el)
                    if not label_text:
                        continue
                    fields.append({"kind": "text", "label": label_text, "name": name, "id": eid, "options": None})
                except Exception:
                    pass
        except Exception:
            pass
        return fields

    def _llm_answer_fields(self, cv_data, fields, page_text):
        """Ask Groq to answer remaining unfilled fields (dropdowns, radios, open text) from the CV."""
        if not GROQ_API_KEY or not fields:
            return []
        try:
            llm = ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-120b", temperature=0, reasoning_effort="low", max_tokens=3000)
            fallback_llm = ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-20b", temperature=0, reasoning_effort="low", max_tokens=3000)
            prompt = f"""You are a job application assistant filling out an external company application form
(not LinkedIn/Indeed's own flow — this is a company careers site or ATS like Greenhouse/Workday/Lensa).

CURRENT PAGE TEXT (for context):
{page_text[:1200]}

VISIBLE UNFILLED FIELDS:
{json.dumps(fields, separators=(',', ':'))}

CANDIDATE CV:
{json.dumps(_lean_cv_for_prompt(cv_data), separators=(',', ':'))}

TASK: Return the correct value for each field based ONLY on the candidate's CV.
Rules:
- For "select"/"radio" fields, you MUST return one of the exact strings listed in that field's "options".
- For "checkbox" fields, return "check" only for standard consent/terms/privacy-policy acknowledgements; otherwise "leave unchecked".
- "years of experience" -> calculate conservatively from CV dates.
- work authorization / eligibility -> pick the option meaning "Yes".
- visa sponsorship required -> pick the option meaning "No".
- "how did you hear about us" -> pick "LinkedIn" or the closest option.
- salary / compensation expectations -> if it's a dropdown/select with an option meaning
  "Negotiable", choose that option; otherwise leave empty ("").
- "willing to relocate" (in any phrasing, including "at your own expense") -> "No".
- "licenses and certifications" -> list them from the CANDIDATE CV if present, otherwise
  answer exactly "N.A.".
- gender / sex / voluntary self-identification -> if one of the options means "Decline to
  self-identify" or "Prefer not to say", choose that one; never guess Male/Female — if no
  such option exists, return "".
- "verify/confirm/re-enter email" -> repeat the exact "email" value from the CANDIDATE CV above.
- "verify/confirm/re-enter phone" -> repeat the exact "phone" value from the CANDIDATE CV above.
- open-text questions -> 1-2 sentences using CV evidence only, never fabricate.
- if a field can't be answered from the CV -> return "".

Return a JSON array ONLY, no explanation, matching each field by "id" (or "name" if it has no id):
[{{"id": "field_id_or_empty", "name": "field_name_or_empty", "value": "answer"}}, ...]"""

            response = retry_groq_call(
                lambda: llm.invoke([HumanMessage(content=prompt)]),
                fallback_fn=lambda: fallback_llm.invoke([HumanMessage(content=prompt)]),
            )
            text = response.content.strip()
            if "[" in text and "]" in text:
                text = text[text.find("["):text.rfind("]") + 1]
            return json.loads(text)
        except Exception as e:
            print(f"  [WARN] LLM form-answer error: {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", f"  [WARN] LLM form-answer error: {e}")
            return []

    def _apply_llm_field_answers(self, answers):
        for item in answers:
            fid = (item.get("id") or "").strip()
            fname = (item.get("name") or "").strip()
            val = item.get("value", "")
            if not val or (not fid and not fname):
                continue
            try:
                el = None
                if fid:
                    matches = self.driver.find_elements(By.CSS_SELECTOR, f"[id='{fid}']")
                    el = matches[0] if matches else None
                if not el and fname:
                    matches = self.driver.find_elements(By.CSS_SELECTOR, f"[name='{fname}']")
                    if matches and (matches[0].get_attribute("type") or "").lower() != "radio":
                        el = matches[0]

                if el and el.is_displayed():
                    tag = el.tag_name.lower()
                    itype = (el.get_attribute("type") or "").lower()
                    if tag == "select":
                        sel_obj = Select(el)
                        try:
                            sel_obj.select_by_visible_text(val)
                            print(f"  [OK] LLM-filled dropdown '{fid or fname}' -> '{val}'")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Generic Filler", f"  [OK] LLM-filled dropdown '{fid or fname}' -> '{val}'")
                        except Exception:
                            for opt in sel_obj.options:
                                if opt.text.strip().lower() == val.strip().lower():
                                    sel_obj.select_by_visible_text(opt.text)
                                    break
                    elif itype == "checkbox":
                        if val.lower() == "check" and not el.is_selected():
                            self.driver.execute_script("arguments[0].click();", el)
                            print(f"  [OK] LLM-checked checkbox '{fid or fname}'")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Generic Filler", f"  [OK] LLM-checked checkbox '{fid or fname}'")
                    else:
                        current = (el.get_attribute("value") or "").strip()
                        if not current:
                            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                            try:
                                el.click()
                            except Exception:
                                pass
                            try:
                                self.driver.execute_script(_NATIVE_SET_VALUE_JS, el, "")
                            except Exception:
                                pass
                            try:
                                el.send_keys(Keys.CONTROL + "a")
                                el.send_keys(Keys.BACKSPACE)
                            except Exception:
                                pass
                            el.send_keys(str(val))
                            final_value = (el.get_attribute("value") or "").strip()
                            if final_value != str(val).strip():
                                try:
                                    self.driver.execute_script(_NATIVE_SET_VALUE_JS, el, str(val))
                                except Exception:
                                    pass
                            print(f"  [OK] LLM-filled field '{fid or fname}' -> '{str(val)[:40]}'")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Generic Filler", f"  [OK] LLM-filled field '{fid or fname}' -> '{str(val)[:40]}'")
                    continue

                # Radio group: no single element id — find the option matching the answer text.
                if fname:
                    radios = self.driver.find_elements(By.CSS_SELECTOR, f"input[type='radio'][name='{fname}']")
                    for r in radios:
                        rid = r.get_attribute("id") or ""
                        label = ""
                        if rid:
                            try:
                                label = self.driver.find_element(By.CSS_SELECTOR, f"label[for='{rid}']").text.strip()
                            except Exception:
                                pass
                        label = label or r.get_attribute("value") or ""
                        if label.strip().lower() == val.strip().lower() and not r.is_selected():
                            self.driver.execute_script("arguments[0].click();", r)
                            print(f"  [OK] LLM-selected radio '{fname}' -> '{val}'")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Generic Filler", f"  [OK] LLM-selected radio '{fname}' -> '{val}'")
                            break
            except Exception:
                pass

    def _page_signature(self):
        try:
            return self.driver.find_element(By.TAG_NAME, "body").text[:3000]
        except Exception:
            return ""

    def _is_final_submit_step(self):
        """True if a final Submit/Apply button (not Next/Continue) is visible on the current step."""
        try:
            selectors = [
                "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'submit')]",
                "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'apply now')]",
                "button[type='submit']",
                "input[type='submit']",
            ]
            for sel in selectors:
                by_type = By.XPATH if sel.startswith("xpath:") else By.CSS_SELECTOR
                actual_sel = sel[6:] if sel.startswith("xpath:") else sel
                for el in self.driver.find_elements(by_type, actual_sel):
                    if el.is_displayed() and el.is_enabled():
                        text = (el.text or el.get_attribute("value") or "").strip().lower()
                        if "next" in text or "continue" in text:
                            continue
                        if any(k in text for k in ("submit", "send application", "apply now")):
                            return True
        except Exception:
            pass
        return False

    def _click_continue_button(self):
        """Click a Next/Continue/Save & Continue button to advance a multi-step form. Never clicks a Submit button."""
        continue_selectors = [
            "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue')]",
            "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'next')]",
            "xpath://a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue')]",
            "button[type='submit']",
        ]
        for sel in continue_selectors:
            try:
                by_type = By.XPATH if sel.startswith("xpath:") else By.CSS_SELECTOR
                actual_sel = sel[6:] if sel.startswith("xpath:") else sel
                elements = self.driver.find_elements(by_type, actual_sel)
                for el in elements:
                    if not (el.is_displayed() and el.is_enabled()):
                        continue
                    text = (el.text or el.get_attribute("value") or "").strip().lower()
                    if "submit" in text and "continue" not in text and "next" not in text:
                        continue
                    self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                    time.sleep(0.2)
                    try:
                        el.click()
                    except Exception:
                        self.driver.execute_script("arguments[0].click();", el)
                    print(f"  [OK] Clicked continue/next: '{text}'")

                    if hasattr(self, "current_job_id") and self.current_job_id:

                        log_step(self.current_job_id, "Generic Filler", f"  [OK] Clicked continue/next: '{text}'")
                    return True
            except Exception:
                pass
        return False

    def fill_form(self, tailored_cv, cover_letter, cv_filepath=None, job_id=None):
        self.current_job_id = job_id
        """
        Fill an external application form. Handles multi-step wizards (e.g. Lensa,
        Workday, Greenhouse) by looping: fill whatever is visible on the current
        step, then click Continue/Next and repeat, until a final Submit step is
        reached (left for the human to review and submit) or the form gets stuck.
        """
        try:
            handles_before_click = self.driver.window_handles
            clicked = self._click_initial_apply_button()

            if clicked:
                # Many "Apply on company site" buttons open a new tab (target="_blank").
                # If we don't switch into it, every subsequent find/fill call below keeps
                # operating on the old tab, which has no form on it at all — silently
                # "filling nothing" even though the click itself worked. LinkedIn's and
                # Indeed's own external-redirect branches already do this before calling
                # fill_form; this covers every other caller (Dice, generic sites) and any
                # second-level redirect those branches don't check for.
                time.sleep(1.0)  # give a same-process popup a moment to register as a new handle
                handles_after_click = self.driver.window_handles
                if len(handles_after_click) > len(handles_before_click):
                    new_tab = [h for h in handles_after_click if h not in handles_before_click][-1]
                    self.driver.switch_to.window(new_tab)
                    print("  [OK] Switched to newly opened tab for the application form.")

                    if hasattr(self, "current_job_id") and self.current_job_id:

                        log_step(self.current_job_id, "Generic Filler", "  [OK] Switched to newly opened tab for the application form.")
                    self.wait_for_cloudflare()

            max_steps = 6
            stuck_count = 0

            for step in range(max_steps):
                identity_filled = self._fill_identity_and_links(tailored_cv)
                if step == 0 and identity_filled == 0:
                    # A slow-hydrating page (heavy JS ATS widgets) can still be rendering
                    # when the first pass runs, landing on zero matches for reasons that
                    # have nothing to do with selector coverage. One short wait-and-retry
                    # rules that out before treating it as a real "unrecognized form".
                    print("  [WARN] No identity fields matched on first pass — waiting and retrying "
                          "once in case the page was still rendering...")
                    time.sleep(2.5)
                    identity_filled = self._fill_identity_and_links(tailored_cv)
                    if identity_filled == 0:
                        print("  [WARN] Still no identity fields (name/email/phone/etc) matched any "
                              "known selector on this page — this site's form may use an unrecognized "
                              "field naming pattern. Falling back to LLM field detection for everything.")

                if cv_filepath and os.path.exists(cv_filepath):
                    self._try_upload_file(cv_filepath, kind="resume")

                uploaded_cl_file = False
                if cover_letter and job_id:
                    cl_path = _generate_cover_letter_docx_file(cover_letter, job_id)
                    uploaded_cl_file = self._try_upload_file(cl_path, kind="cover_letter")
                if cover_letter and not uploaded_cl_file:
                    self._try_fill_any([
                        "textarea[name*='cover' i]", "textarea[id*='cover' i]",
                        "textarea[name*='letter' i]", "textarea[id*='letter' i]",
                        "textarea[name*='motivation' i]", "textarea[id*='motivation' i]",
                        "textarea[placeholder*='cover letter' i]",
                    ], cover_letter)

                self._click_consent_checkboxes()
                self._fill_known_dropdowns()
                self._fill_known_radio_groups()
                self._fill_date_inputs()
                self._fill_range_sliders()

                page_text = self._page_signature()
                remaining_fields = self._collect_unfilled_fields()
                if remaining_fields:
                    answers = self._llm_answer_fields(tailored_cv, remaining_fields, page_text)
                    if answers:
                        self._apply_llm_field_answers(answers)
                        print(f"  [OK] LLM answered {len(answers)} additional field(s)")

                        if hasattr(self, "current_job_id") and self.current_job_id:

                            log_step(self.current_job_id, "Generic Filler", f"  [OK] LLM answered {len(answers)} additional field(s)")

                if self._is_final_submit_step():
                    print("  [OK] Form fully filled — reached final submit step. Waiting for your review/submission.")

                    if hasattr(self, "current_job_id") and self.current_job_id:

                        log_step(self.current_job_id, "Generic Filler", "  [OK] Form fully filled — reached final submit step. Waiting for your review/submission.")
                    return True

                text_before = self._page_signature()
                clicked = self._click_continue_button()
                if not clicked:
                    print("  [OK] No further Continue/Next button found — treating current step as final.")

                    if hasattr(self, "current_job_id") and self.current_job_id:

                        log_step(self.current_job_id, "Generic Filler", "  [OK] No further Continue/Next button found — treating current step as final.")
                    return True

                time.sleep(1.8)
                self.wait_for_cloudflare()

                if self._page_signature() == text_before:
                    stuck_count += 1
                else:
                    stuck_count = 0

                if stuck_count >= 2:
                    print("  [WARN] Continue click had no visible effect after repeated attempts — leaving form for manual completion.")

                    if hasattr(self, "current_job_id") and self.current_job_id:

                        log_step(self.current_job_id, "Generic Filler", "  [WARN] Continue click had no visible effect after repeated attempts — leaving form for manual completion.")
                    return True

            print("  [WARN] Reached max auto-fill steps — leaving form for manual completion/review.")


            if hasattr(self, "current_job_id") and self.current_job_id:


                log_step(self.current_job_id, "Generic Filler", "  [WARN] Reached max auto-fill steps — leaving form for manual completion/review.")
            return True
        except Exception as e:
            print(f"Error during auto-fill: {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", f"Error during auto-fill: {e}")
            return False

    def wait_for_cloudflare(self):
        """Wait if a Cloudflare challenge is detected on the current page."""
        try:
            print("Checking for Cloudflare challenge...")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", "Checking for Cloudflare challenge...")
            for i in range(30):
                title = self.driver.title
                cf_titles = ("Just a moment", "Attention Required", "Security check", "Cloudflare")
                if any(t in title for t in cf_titles):
                    print(f"  Cloudflare detected (attempt {i+1}). Waiting for you to solve it...")

                    if hasattr(self, "current_job_id") and self.current_job_id:

                        log_step(self.current_job_id, "Generic Filler", f"  Cloudflare detected (attempt {i+1}). Waiting for you to solve it...")
                    if i in (5, 15):
                        self.driver.refresh()
                    time.sleep(2)
                else:
                    elements = self.driver.find_elements(
                        By.CSS_SELECTOR,
                        "#challenge-running, #cf-challenge-running, .cf-browser-verification"
                    )
                    if elements:
                        print(f"  Cloudflare element detected (attempt {i+1}). Waiting...")

                        if hasattr(self, "current_job_id") and self.current_job_id:

                            log_step(self.current_job_id, "Generic Filler", f"  Cloudflare element detected (attempt {i+1}). Waiting...")
                        if i in (5, 15):
                            self.driver.refresh()
                        time.sleep(2)
                    else:
                        return True
            print("  Cloudflare did not resolve within timeout.")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", "  Cloudflare did not resolve within timeout.")
        except Exception as e:
            print(f"  Cloudflare check error: {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", f"  Cloudflare check error: {e}")
        return False

    def wait_for_cloudflare_on_url(self, url: str):
        """Open a URL then wait for any Cloudflare challenge to clear."""
        self.driver.get(url)
        time.sleep(3)
        self.wait_for_cloudflare()
        time.sleep(1)

    def _close_extra_job_windows(self, original_handles):
        """
        Closes every browser tab/window opened during this job's application
        flow, leaving only the original tab open — used when the user presses
        Skip, per explicit request: a skipped job shouldn't leave stray
        external-apply tabs open in the background. Closes ALL non-original
        handles (not just the current one), since a job can leave more than
        one extra tab behind. Safe to call even if the browser is already
        partially/fully closed (e.g. it raced with the "browser closed"
        detection in apply_to_job) — every window-handle operation here is
        wrapped defensively so this never raises.
        """
        try:
            for handle in self.driver.window_handles:
                if handle not in original_handles:
                    try:
                        self.driver.switch_to.window(handle)
                        self.driver.close()
                    except Exception:
                        pass
            remaining = self.driver.window_handles
            if remaining:
                target = original_handles[0] if original_handles[0] in remaining else remaining[0]
                self.driver.switch_to.window(target)
        except Exception:
            pass

    def apply_to_job(self, job, _crash_retry=False):
        self.current_job_id = str(job.get('id') or job.get('job_id'))
        job_id = str(job.get('id') or job.get('job_id'))
        job_title = job.get('job_title', 'Unknown Title')
        company = job.get('company', 'Unknown Company')
        job_url = job.get('url') or job.get('job_url')
        job_source = (job.get('source') or '').lower()

        # The 180s countdown starts the instant this job begins processing — i.e. the
        # moment "Apply for this job" is clicked — and covers the bot's own fill-in
        # work as well as the human review window as one continuous budget. This is
        # what the dashboard's countdown bar reads via /apply/status the whole time.
        review_deadline_epoch["value"] = time.time() + REVIEW_TIMEOUT_SECONDS
        awaiting_review["value"] = False

        review_event.clear()
        skip_event.clear()

        if not job_url:
            log_application(job_id, job_title, company, JobStatus.FAILED, "Missing application URL")
            review_deadline_epoch["value"] = None
            return False

        try:
            tailored_data = _load_tailored_for_job(job_id)

            if tailored_data:
                tailored_cv = tailored_data.get("cv", {})
                cover_letter = tailored_data.get("cover_letter", "")
                print(f"Using pre-generated tailored CV and cover letter for {job_title}")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"Using pre-generated tailored CV and cover letter for {job_title}")
            else:
                print(f"Generating cover letter for {job_title} at {company}...")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"Generating cover letter for {job_title} at {company}...")
                job_desc = job.get('description', '')
                # CV is used as-is (no LLM rewriting) — only the cover letter is
                # generated per job description, per explicit instruction to stop
                # tailoring the CV.
                tailored_cv = self.cv_data
                cover_letter = self.cover_letter_func(self.cv_data, job_desc)
                save_tailored_application(job_id, {
                    "cv": tailored_cv,
                    "cv_latex": render_tailored_cv_latex(tailored_cv),
                    "cover_letter": cover_letter
                })

            log_application(job_id, job_title, company, JobStatus.TAILORED)
            save_job_snapshot(job_id, job)

            # Prioritize the originally uploaded high-quality PDF/DOCX resume
            original_resume_path = _get_latest_uploaded_resume()
            if original_resume_path:
                cv_filepath = original_resume_path
                print(f"Using high-quality original resume for upload: {os.path.basename(cv_filepath)}")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"Using high-quality original resume for upload: {os.path.basename(cv_filepath)}")
            else:
                cv_filepath = _generate_cv_text_file(tailored_cv, job_id)
                print(f"Fallback to text tailored resume: {os.path.basename(cv_filepath)}")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"Fallback to text tailored resume: {os.path.basename(cv_filepath)}")

            self._jobs_since_restart += 1
            if self._jobs_since_restart > self.RESTART_EVERY_N_JOBS and self.driver:
                print(f"[INFO] Recycling browser session after {self.RESTART_EVERY_N_JOBS} applications...")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"[INFO] Recycling browser session after {self.RESTART_EVERY_N_JOBS} applications...")
                try:
                    self.driver.quit()
                except Exception:
                    pass
                self.driver = None
                self._jobs_since_restart = 1

            self.init_browser()
            self.wait_for_cloudflare_on_url(job_url)

            # Record window handles to manage external redirections/new tabs
            original_handles = self.driver.window_handles
            main_handle = self.driver.current_window_handle

            # ── Route by job source ───────────────────────────────────────────
            if job_source == 'linkedin' or 'linkedin.com' in job_url.lower():
                print(f"[LinkedIn Agent] Starting LangGraph agent for: {job_title} at {company}")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"[LinkedIn Agent] Starting LangGraph agent for: {job_title} at {company}")
                result = run_linkedin_agent(
                    driver=self.driver,
                    job_url=job_url,
                    cv_data=tailored_cv,
                    cover_letter=cover_letter,
                    cv_filepath=cv_filepath,
                    job_id=job_id,
                )
                if result == 'already_applied':
                    log_application(job_id, job_title, company, JobStatus.SUBMITTED, "Already applied via LinkedIn")
                    review_deadline_epoch["value"] = None
                    return True
                elif result == 'failed':
                    print(f"[LinkedIn Agent] Could not complete — leaving browser open for manual apply")

                    if hasattr(self, "current_job_id") and self.current_job_id:

                        log_step(self.current_job_id, "Generic Filler", f"[LinkedIn Agent] Could not complete — leaving browser open for manual apply")

            elif job_source == 'indeed' or _is_indeed_url(job_url):
                print(f"[Indeed Agent] Starting for: {job_title} at {company}")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"[Indeed Agent] Starting for: {job_title} at {company}")
                agent = IndeedApplier(
                    driver=self.driver,
                    cv_data=tailored_cv,
                    cover_letter=cover_letter,
                    cv_filepath=cv_filepath,
                    job_id=job_id,
                )
                result = agent.run(job_url, job_id=job_id)
                if result == 'failed':
                    print(f"[Indeed Agent] Could not reach review page — leaving for human review")

                    if hasattr(self, "current_job_id") and self.current_job_id:

                        log_step(self.current_job_id, "Generic Filler", f"[Indeed Agent] Could not reach review page — leaving for human review")

            elif job_source == 'dice' or 'dice.com' in job_url.lower():
                # Dormant: Dice discovery is switched off (see discover.py), so no new
                # jobs route here — kept in place only so an already-tracked Dice job
                # already in the pipeline can still be applied to.
                from src.workflow.dice_apply import DiceApplier
                print(f"[Dice Agent] Starting for: {job_title} at {company}")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"[Dice Agent] Starting for: {job_title} at {company}")
                agent = DiceApplier(
                    driver=self.driver,
                    cv_data=tailored_cv,
                    cover_letter=cover_letter,
                    cv_filepath=cv_filepath,
                    job_id=job_id,
                )
                agent.run(job_url, job_id=job_id)

            elif job_source == 'builtin' or 'builtin.com' in job_url.lower():
                from src.workflow.builtin_apply import BuiltInApplier
                print(f"[BuiltIn Agent] Starting for: {job_title} at {company}")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"[BuiltIn Agent] Starting for: {job_title} at {company}")
                agent = BuiltInApplier(
                    driver=self.driver,
                    cv_data=tailored_cv,
                    cover_letter=cover_letter,
                    cv_filepath=cv_filepath,
                    job_id=job_id,
                )
                agent.run(job_url, job_id=job_id, apply_url=job.get('apply_url'))

            else:
                # Generic path: open URL and fill form directly
                print(f"[Generic Filler] Filling form for: {job_title} at {company}")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"[Generic Filler] Filling form for: {job_title} at {company}")
                self.wait_for_cloudflare()
                self.fill_form(tailored_cv, cover_letter, cv_filepath, job_id=job_id)

            log_application(job_id, job_title, company, JobStatus.READY)

            print(f"--- [Job {job_id}] REVIEW REQUIRED ---")


            if hasattr(self, "current_job_id") and self.current_job_id:


                log_step(self.current_job_id, "Generic Filler", f"--- [Job {job_id}] REVIEW REQUIRED ---")
            print(f"Role: {job_title} | Company: {company}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", f"Role: {job_title} | Company: {company}")
            print(f"Action: Review the form in the browser, then click 'Submit Done' in the dashboard.")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", f"Action: Review the form in the browser, then click 'Submit Done' in the dashboard.")

            try:
                js_banner = """
                if (!document.getElementById('__jobbot_review_banner')) {
                    var b = document.createElement('div');
                    b.id = '__jobbot_review_banner';
                    b.style.cssText = 'position:fixed;top:0;left:0;width:100%;background-color:#007aff;color:white;text-align:center;padding:12px;font-family:sans-serif;font-size:16px;font-weight:bold;z-index:999999;box-shadow:0 2px 10px rgba(0,0,0,0.2);';
                    b.innerText = 'Bot paused — please review or complete the form manually, then click Submit Done in the dashboard.';
                    document.body.appendChild(b);
                }
                """
                self.driver.execute_script(js_banner)
            except Exception:
                pass

            # Only now — with the form actually ready for you — does the
            # dashboard switch from "bot is filling" to the Skip/Submit
            # banner, and only now does the review clock start. Previously
            # the same 180s budget set at the very top of this function had
            # to cover the bot's own fill time too, so a job that struggled
            # for a few minutes (like the QTech US Inc case) could already
            # be down to 0:00 by the time a real review ever started.
            awaiting_review["value"] = True
            review_deadline_epoch["value"] = time.time() + REVIEW_TIMEOUT_SECONDS

            timed_out = False

            # Wait for human signal, auto-detect, or the review timeout
            while not review_event.is_set() and not skip_event.is_set():
                if time.time() >= review_deadline_epoch["value"]:
                    timed_out = True
                    break
                try:
                    current_url = self.driver.current_url.lower()

                    # 1. Check for expired linkedin job
                    if 'linkedin.com' in current_url:
                        try:
                            body_el = self.driver.find_element(By.TAG_NAME, "body")
                            if body_el and "no longer accepting applications" in body_el.text.lower():
                                print(f"LinkedIn job expired for: {job_title}")

                                if hasattr(self, "current_job_id") and self.current_job_id:

                                    log_step(self.current_job_id, "Generic Filler", f"LinkedIn job expired for: {job_title}")
                                log_application(job_id, job_title, company, JobStatus.EXPIRED, "No longer accepting applications")
                                review_deadline_epoch["value"] = None
                                awaiting_review["value"] = False
                                try:
                                    self.driver.close()
                                    if self.driver.window_handles:
                                        self.driver.switch_to.window(self.driver.window_handles[0])
                                except Exception:
                                    pass
                                return False
                        except Exception:
                            pass

                    # 2. Check for manual submission detection
                    is_submitted = False

                    # URL change detection — match on the URL *path/query*
                    # only (never the hostname), with word boundaries. Many
                    # real ATS domains legitimately contain these words as
                    # substrings (e.g. successfactors.com), which was
                    # false-triggering "submitted" the instant the form
                    # loaded, before the user had done anything.
                    try:
                        parsed_url = urlparse(current_url)
                        url_path_and_query = (parsed_url.path + "?" + parsed_url.query).lower()
                    except Exception:
                        url_path_and_query = ""
                    url_looks_like_confirmation = bool(re.search(
                        r'\b(success|thank[-_]?you|completed|submitted|post-apply|confirmation)\b',
                        url_path_and_query
                    ))

                    if url_looks_like_confirmation:
                        try:
                            body_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
                        except Exception:
                            body_text = ""
                        # A matching URL alone isn't enough corroboration —
                        # require the page content to actually confirm a
                        # submission happened.
                        if any(phrase in body_text for phrase in (
                            "application has been submitted", "application was submitted",
                            "successfully submitted", "thank you for applying",
                            "thank you for your application", "application received",
                            "your application is complete",
                        )):
                            is_submitted = True

                    if not is_submitted:
                        try:
                            # JS event listener check — only counts a genuine
                            # final-submit click (exact "Submit" / "Submit
                            # Application" text), not any element that merely
                            # contains the word "submit" (e.g. "Submit and
                            # Continue" wizard buttons, "Resubmit", "Submit
                            # availability"), since those fire mid-flow on
                            # multi-step forms and were closing the review out
                            # from under the user before the real submission.
                            #
                            # e.target is walked up to the nearest button-like
                            # ancestor via closest() before reading its text —
                            # confirmed live this was the actual reason Workday's
                            # real Submit clicks were never detected: Workday
                            # renders it as a <div role="button"> wrapping a
                            # separate icon element and a text span (the same
                            # pattern already confirmed on its "Save and
                            # Continue" button), so a click landing on the icon
                            # or padding instead of squarely on the text made
                            # e.target some sub-element with no text at all,
                            # and the regex never matched — verified with an
                            # isolated fixture: the old check missed an icon-area
                            # click 100% of the time, the closest()-based one
                            # caught it every time. Simple <button>Submit</button>
                            # ATS's (Greenhouse, Ashby, etc.) are unaffected
                            # either way since closest() on a plain button just
                            # returns itself.
                            js_detector = """
                            if (!window.__bot_detector_injected) {
                                window.__bot_detector_injected = true;
                                window.__bot_form_submitted = false;
                                document.addEventListener('click', function(e) {
                                    let target = e.target.closest("button, [role='button'], input[type='submit'], input[type='button'], a") || e.target;
                                    let t = (target.innerText || target.value || target.getAttribute('aria-label') || '').trim().toLowerCase();
                                    if (/^(submit|submit application|submit your application)$/.test(t)) {
                                        window.__bot_form_submitted = true;
                                    }
                                }, true);
                                document.addEventListener('submit', function(e) {
                                    window.__bot_form_submitted = true;
                                }, true);
                            }
                            return window.__bot_form_submitted;
                            """
                            if self.driver.execute_script(js_detector):
                                # A submit-like click alone isn't proof the submission
                                # actually went through — confirmed live on a Rhombus
                                # Power / Greenhouse job: the form still had ~13
                                # unanswered/invalid custom dropdown fields when the
                                # click happened, Greenhouse's own client-side
                                # validation blocked the real submission and showed
                                # inline errors instead, but the click event itself
                                # still fired — the old code trusted it immediately,
                                # closed the browser, and wrongly marked the job
                                # Submitted even though nothing was actually sent.
                                # Now waits for the page to react, then requires no
                                # visible validation-error indicators still on screen
                                # before trusting it. If errors are still showing, the
                                # click didn't actually go through — re-arm the
                                # listener and keep waiting for a genuine one instead
                                # of finalizing.
                                print(f"Submit-like click detected for: {job_title} — verifying it actually went through...")

                                if hasattr(self, "current_job_id") and self.current_job_id:

                                    log_step(self.current_job_id, "Generic Filler", f"Submit-like click detected for: {job_title} — verifying it actually went through...")

                                time.sleep(5)

                                # Let a dead browser raise naturally here so it falls
                                # through to the existing skip/submit-race-aware
                                # "browser closed" handling in the except block below,
                                # instead of this wait silently treating a browser that
                                # died during it as a confirmed submission.
                                self.driver.title

                                submission_confirmed = True
                                try:
                                    error_check_js = """
                                    var selectors = ["[aria-invalid='true']", ".error, .field-error, .has-error, .invalid-feedback, .is-invalid", "[class*='error' i]:not(script):not(style)"];
                                    for (var s = 0; s < selectors.length; s++) {
                                        var els = document.querySelectorAll(selectors[s]);
                                        for (var i = 0; i < els.length; i++) {
                                            var el = els[i];
                                            var rect = el.getBoundingClientRect();
                                            var visible = rect.width > 0 && rect.height > 0 && window.getComputedStyle(el).visibility !== 'hidden';
                                            if (visible && (el.innerText || '').trim().length > 0) {
                                                return true;
                                            }
                                        }
                                    }
                                    return false;
                                    """
                                    if self.driver.execute_script(error_check_js):
                                        submission_confirmed = False
                                except Exception:
                                    pass

                                if submission_confirmed:
                                    is_submitted = True
                                else:
                                    print(f"Submit click did not actually go through for: {job_title} (validation errors still visible) — resuming wait.")

                                    if hasattr(self, "current_job_id") and self.current_job_id:

                                        log_step(self.current_job_id, "Generic Filler", f"Submit click did not actually go through for: {job_title} (validation errors still visible) — resuming wait.")
                                    try:
                                        self.driver.execute_script("window.__bot_form_submitted = false;")
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                    if is_submitted:
                        print(f"Auto-detected submission for: {job_title}")

                        if hasattr(self, "current_job_id") and self.current_job_id:

                            log_step(self.current_job_id, "Generic Filler", f"Auto-detected submission for: {job_title}")
                        review_event.set()
                        break

                    # Standard check to ensure browser is still open
                    self.driver.title
                except Exception as e:
                    review_deadline_epoch["value"] = None
                    awaiting_review["value"] = False
                    # A Skip or Submit click can race with this check — if either
                    # signal was already set by the time the browser became
                    # unreachable, that's what actually happened, not "you closed
                    # the browser without submitting." Confirmed live on a
                    # BankUnited "Cloud Data Engineer II" job: the generic
                    # "browser closed" branch below used to win this race
                    # unconditionally, silently swallowing a genuine Skip click and
                    # finalizing the job as TAILORED/"User closed browser" instead
                    # of SKIPPED — so it never showed up in today's Application
                    # Pipeline the way a real skip should. Checked here first so a
                    # Skip/Submit that lands right as the browser check fails is
                    # never lost.
                    if skip_event.is_set():
                        print(f"Skipped: {job_title}")

                        if hasattr(self, "current_job_id") and self.current_job_id:

                            log_step(self.current_job_id, "Generic Filler", f"Skipped: {job_title}")
                        log_application(job_id, job_title, company, JobStatus.SKIPPED, "User skipped during review")
                        self._close_extra_job_windows(original_handles)
                        return False
                    if review_event.is_set():
                        print(f"[OK] Submitted: {job_title}")

                        if hasattr(self, "current_job_id") and self.current_job_id:

                            log_step(self.current_job_id, "Generic Filler", f"[OK] Submitted: {job_title}")
                        log_application(job_id, job_title, company, JobStatus.SUBMITTED)
                        return False
                    print(f"Browser closed for: {job_title}")

                    if hasattr(self, "current_job_id") and self.current_job_id:

                        log_step(self.current_job_id, "Generic Filler", f"Browser closed for: {job_title}")
                    log_application(job_id, job_title, company, JobStatus.TAILORED, "User closed browser before submit")
                    return False
                time.sleep(1.5)

            review_deadline_epoch["value"] = None
            awaiting_review["value"] = False

            if timed_out:
                print(f"Review timed out after {REVIEW_TIMEOUT_SECONDS}s without submission: {job_title}")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"Review timed out after {REVIEW_TIMEOUT_SECONDS}s without submission: {job_title}")
                log_application(job_id, job_title, company, JobStatus.SKIPPED, f"Timed out after {REVIEW_TIMEOUT_SECONDS}s without submission")
                try:
                    self.driver.quit()
                except Exception:
                    pass
                self.driver = None
                return False

            if skip_event.is_set():
                print(f"Skipped: {job_title}")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"Skipped: {job_title}")
                log_application(job_id, job_title, company, JobStatus.SKIPPED, "User skipped during review")
                self._close_extra_job_windows(original_handles)
                return False

            log_application(job_id, job_title, company, JobStatus.SUBMITTED)
            print(f"[OK] Submitted: {job_title}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", f"[OK] Submitted: {job_title}")

            # Clean up: close secondary tabs and return to primary handle
            try:
                current_handle = self.driver.current_window_handle
                if len(self.driver.window_handles) > 1 and current_handle != original_handles[0]:
                    self.driver.close()
                    self.driver.switch_to.window(original_handles[0])
                else:
                    self.driver.get("about:blank")
            except Exception:
                pass

            return True

        except SkipRequested:
            # A Skip click while the bot was still stuck mid-fill (e.g. inside
            # a native agent's own wait loop) — handled exactly like a skip
            # requested during the normal review wait: SKIPPED status, tabs
            # closed, back to overview. Must be caught here, before the
            # generic `except Exception` below, which would otherwise
            # mislabel this as a FAILED job.
            print(f"Skipped (while bot was still working): {job_title}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", f"Skipped (while bot was still working): {job_title}")
            log_application(job_id, job_title, company, JobStatus.SKIPPED, "User skipped while bot was still filling the form")
            self._close_extra_job_windows(original_handles)
            review_deadline_epoch["value"] = None
            awaiting_review["value"] = False
            return False

        except BrowserClosedByUser:
            # The browser window closed while the bot was waiting for you to
            # manually complete a step it couldn't handle itself — see
            # BrowserClosedByUser's docstring for the false-Submitted /
            # infinite-hang bugs this replaces. Treated exactly like closing
            # the browser during the final review step already is: an honest
            # "not submitted" stop, not a retry, so it lands back in today's
            # Application Pipeline for you to revisit.
            print(f"Browser closed for: {job_title}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", f"Browser closed for: {job_title}")
            log_application(job_id, job_title, company, JobStatus.TAILORED, "User closed browser before submit")
            review_deadline_epoch["value"] = None
            awaiting_review["value"] = False
            return False

        except BrowserCrashed as e:
            # The entire browser/ChromeDriver process is gone — confirmed
            # live on a Tata Consultancy Services job where every Selenium
            # call started failing with a connection-refused error and the
            # bot spent 11+ minutes blindly retrying selectors against a
            # connection that could never come back. A dead tab is
            # recoverable (existing _recover_dead_window() logic); a dead
            # connection isn't, so this relaunches a completely fresh
            # browser and retries the SAME job once from scratch before
            # giving up — rather than requiring you to notice and restart
            # everything by hand.
            review_deadline_epoch["value"] = None
            awaiting_review["value"] = False
            try:
                if self.driver:
                    self.driver.quit()
            except Exception:
                pass
            self.driver = None

            if _crash_retry:
                print(f"Browser crashed again while applying to {job_title} — giving up after one retry: {e}")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Generic Filler", f"Browser crashed again while applying to {job_title} — giving up after one retry: {e}")
                log_application(job_id, job_title, company, JobStatus.FAILED, f"Browser crashed twice — please retry manually: {e}")
                return False

            print(f"Browser crashed while applying to {job_title} — relaunching a fresh browser and retrying once: {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", f"Browser crashed while applying to {job_title} — relaunching a fresh browser and retrying once: {e}")
            return self.apply_to_job(job, _crash_retry=True)

        except Exception as e:
            print(f"Failed to apply to {job_title}: {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Generic Filler", f"Failed to apply to {job_title}: {e}")
            log_application(job_id, job_title, company, JobStatus.FAILED, str(e))
            review_deadline_epoch["value"] = None
            awaiting_review["value"] = False
            return False

    def close(self):
        if self.driver:
            self.driver.quit()

def bulk_apply(jobs, cv_data, cover_letter_func):
    applier = SequentialApplier(cv_data, cover_letter_func)
    try:
        for job in jobs:
            applier.apply_to_job(job)
    finally:
        applier.close()

def signal_review_complete():
    review_event.set()

def signal_skip():
    skip_event.set()
