import json
import os
import re
import secrets
import string
import time
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
from src.storage.tracker import log_application, save_tailored_application, JobStatus, APPLICATIONS_DIR, UPLOADS_DIR
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

# Custom-styled dropdown/radio/toggle widgets built from generic elements with
# ARIA roles instead of native <select>/<input radio> — invisible to a plain
# native-element scan, but common on modern React-built ATS forms.
CUSTOM_WIDGET_ROLE_SELECTOR = (
    "[role='combobox']:not(select), [role='listbox']:not(select), "
    "[aria-haspopup='listbox']:not(select), [role='radiogroup'], [role='switch']"
)

# Used by _detect_validation_errors() to tell "this step needs your help"
# (a click redrew the same step with error banners) apart from "this step
# genuinely advanced" — mirrors linkedin_apply.py's _ERROR_SELECTORS.
_ERROR_SELECTORS = [
    "[aria-invalid='true']",
    ".error, .field-error, .has-error, .invalid-feedback, .is-invalid",
    "[class*='error' i]:not(script):not(style)",
]

# Epoch time the current review-wait period expires, or None when no job is
# currently waiting on human review. A plain dict (not a module-level float) so
# the API layer can import and read it live without a getter function.
review_deadline_epoch = {"value": None}


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


class DiceApplier:
    def __init__(self, driver, cv_data: dict, cover_letter: str = "", cv_filepath: str = None, job_id: str = None):
        self.cv_data = cv_data
        self.cover_letter_func = None
        self.driver = driver
        self.current_job_id = None
        self.cover_letter = cover_letter
        self.cv_filepath = cv_filepath

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

                        log_step(self.current_job_id, "Dice Agent", f"  [OK] Filled field: '{selector}' -> '{value}'")
                    return True
        except Exception as e:
            print(f"  [DEBUG] Error filling selector '{selector}': {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Dice Agent", f"  [DEBUG] Error filling selector '{selector}': {e}")
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

                        log_step(self.current_job_id, "Dice Agent", f"  [OK] Uploaded {kind} file via: {selector}")
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

                        log_step(self.current_job_id, "Dice Agent", f"  [OK] Uploaded {kind} file via context-matched input (nearby text: '{own_keyword}')")
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

                            log_step(self.current_job_id, "Dice Agent", f"  [OK] Uploaded {kind} file via: {selector}")
                        self._mark_uploaded(el)
                        time.sleep(1.5)
                        return True

            # Last resort: a plain, single, generic file input with nothing else to go on.
            # Resume fields consistently render before cover-letter fields in DOM order, so
            # calling this for "resume" first and "cover_letter" second naturally targets
            # the right one either way — except when the only candidate is clearly the
            # OTHER kind of document, in either direction — confirmed live on an OpenAI
            # LinkedIn-external-apply job (same underlying pattern applies here): this
            # exclusion used to only protect the resume-upload direction, so a
            # cover-letter upload with nothing labeled "cover" fell through and landed
            # on the visually-obvious Resume box instead, since the real resume upload
            # had already gone to a different, less obvious input matched above.
            all_file_inputs = _unfilled(self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']"))
            exclude_keyword = "resume" if kind == "cover_letter" else "cover"
            all_file_inputs = [el for el in all_file_inputs if exclude_keyword not in self._nearby_container_text(el)]
            if all_file_inputs:
                all_file_inputs[0].send_keys(os.path.abspath(filepath))
                print(f"  [OK] Uploaded {kind} file via first unfilled file input")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Dice Agent", f"  [OK] Uploaded {kind} file via first unfilled file input")
                self._mark_uploaded(all_file_inputs[0])
                time.sleep(1.5)
                return True

            # Nothing matched at all, and not for lack of trying — confirmed live on a
            # New York Power Authority job reached via LinkedIn's external-apply path
            # (same underlying pattern applies here): a required Resume field sat empty
            # because no input[type='file'] existed anywhere on the page reachable by
            # any strategy above, and this returned False completely silently every
            # single pass, giving no indication anything was even attempted.
            if not self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']"):
                print(f"  [WARN] No file input found on the page for {kind} upload — this form may need it uploaded manually.")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Dice Agent", f"  [WARN] No file input found on the page for {kind} upload — this form may need it uploaded manually.")

        except Exception as e:
            print(f"  [WARN] File upload failed ({kind}): {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Dice Agent", f"  [WARN] File upload failed ({kind}): {e}")
        return False

    # Strong signal: a real application form almost always has a resume/file
    # upload or a clearly-named first/last-name/phone field. A job-alert/search/
    # newsletter widget (confirmed live on a Randstad-hosted Dice job) never
    # does — checking this first avoids mistaking such a widget's few generically-
    # named fields for the real form.
    _APPLICATION_IDENTITY_SELECTOR = (
        "input[type='file'], "
        "input[name*='first' i], input[id*='first' i], "
        "input[name*='last' i], input[id*='last' i], "
        "input[name*='resume' i], input[id*='resume' i], "
        "input[name*='phone' i], input[id*='phone' i], input[type='tel']"
    )
    _NON_APPLICATION_FIELD_KEYWORDS = (
        "search", "alert", "subscribe", "newsletter", "query", "frequency", "vendor", "keyword", "signup",
    )
    _FORM_FIELD_SIGNAL_SELECTOR = (
        "input[type='email'], input[type='tel'], input[type='file'], textarea, select, "
        "input[type='text'], input:not([type]), " + CUSTOM_WIDGET_ROLE_SELECTOR
    )
    _FORM_FIELD_MIN_COUNT = 3

    def _frame_has_form_content(self):
        """
        True if the CURRENT frame (top-level page, or whatever iframe Selenium is
        focused on) looks like it contains a real application form. Used by
        _click_initial_apply_button() below (replacing its old single-generic-
        field check, which a job-alert/search widget could satisfy on its own)
        and by _switch_into_form_iframe_if_needed().
        """
        try:
            if any(el.is_displayed() for el in self.driver.find_elements(By.CSS_SELECTOR, self._APPLICATION_IDENTITY_SELECTOR)):
                return True
            candidates = self.driver.find_elements(By.CSS_SELECTOR, self._FORM_FIELD_SIGNAL_SELECTOR)
            filtered = []
            for el in candidates:
                try:
                    if not el.is_displayed():
                        continue
                    haystack = " ".join([
                        el.get_attribute("name") or "", el.get_attribute("id") or "",
                        el.get_attribute("placeholder") or "", el.get_attribute("aria-label") or "",
                    ]).lower()
                    if any(k in haystack for k in self._NON_APPLICATION_FIELD_KEYWORDS):
                        continue
                    filtered.append(el)
                except Exception:
                    continue
            return len(filtered) >= self._FORM_FIELD_MIN_COUNT
        except Exception:
            return False

    def _switch_into_form_iframe_if_needed(self):
        """
        Some external ATS forms reached via a Dice redirect are embedded inside an
        <iframe> on the company's own page rather than served directly (the same
        pattern confirmed live on a Fastly/Greenhouse job) — every fill method in
        this class only ever looks at whatever frame Selenium is currently focused
        on, so such a form would otherwise be completely invisible. Only checks
        top-level iframes, and only switches if the top-level page doesn't already
        have real form content, so this is a no-op on every page that already
        works today (including Dice's own wizard, which is never iframe-embedded).
        """
        if self._frame_has_form_content():
            return True

        try:
            candidates = self.driver.find_elements(By.TAG_NAME, "iframe")
        except Exception:
            candidates = []

        for frame_el in candidates:
            try:
                if not frame_el.is_displayed():
                    continue
                size = frame_el.size
                if size.get("width", 0) < 200 or size.get("height", 0) < 200:
                    continue
                # Must read this BEFORE switching — frame_el belongs to the parent
                # frame's context and becomes invalid the instant switch_to.frame()
                # moves focus into it.
                frame_id = frame_el.get_attribute("id") or ""
            except Exception:
                continue

            try:
                self.driver.switch_to.frame(frame_el)
            except Exception:
                self.driver.switch_to.default_content()
                continue

            if self._frame_has_form_content():
                print(f"  [OK] Switched into an <iframe> containing the real application form (id='{frame_id}')")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Dice Agent", f"  [OK] Switched into an <iframe> containing the real application form (id='{frame_id}')")
                return True

            self.driver.switch_to.default_content()

        print("  [WARN] No form content found on the page or in any embedded iframe.")

        if hasattr(self, "current_job_id") and self.current_job_id:

            log_step(self.current_job_id, "Dice Agent", "  [WARN] No form content found on the page or in any embedded iframe.")
        return False

    def _recover_dead_window(self):
        """
        True if the driver was successfully switched to a live tab after its
        current one was closed out from under it. Confirmed live on a Randstad
        Digital job: Dice's "Apply" opens an intermediate redirector tab, which
        itself opens the REAL application in a further new tab and then closes
        itself a moment later — every operation on the now-dead handle then fails
        with "no such window: target window already closed", including the
        apply-button search, the Cloudflare check, and every single field-fill
        selector, which cascaded into a full auto-fill crash. Returns False if
        the current tab is still fine (nothing to recover) or if no tabs remain.
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

                log_step(self.current_job_id, "Dice Agent", "  [OK] Previous tab closed itself — switched to the new active tab.")
            return True
        except Exception:
            return False

    def _click_initial_apply_button(self):
        """Locate and click the initial 'Apply' / 'Apply Now' button on an external site if the form is not visible yet.

        Even when form-like fields already appear visible, a distinct Apply
        button might ALSO be present on the same page — confirmed live on a
        SIERTEK LTD job reached via LinkedIn's external-apply redirect (same
        underlying pattern applies here since this method mirrors
        linkedin_apply.py's _click_landing_page_apply_if_needed()): an email +
        city + keyword "quick apply/search" widget sat right next to the real
        Apply button and, on its own, satisfied _frame_has_form_content(), so
        the bot filled those unrelated boxes and never clicked the real Apply
        button at all. A single, non-looping probe below (not the full 15s
        retrying search further down) checks for that possibility without
        paying any extra cost on the far more common jobs that land directly
        on the real form with no separate Apply button at all.
        """
        self._initial_apply_click_happened = False
        already_has_form = False
        try:
            already_has_form = self._frame_has_form_content()
        except Exception:
            pass

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

            // Priority 1: explicit "manual"/"direct" apply phrasing, checked before
            // the generic "apply" match below — confirmed live on a LinkedIn/Workday
            // job offering "Apply with LinkedIn" / "Apply with Indeed" / "Apply
            // Manually" together on one page: matching generic "apply" text alone
            // just grabbed whichever button happened to appear first in the page's
            // HTML (an SSO-integrated option the bot can never actually complete),
            // instead of the manual option that actually leads to a fillable form.
            let result = tryClick(h => h.includes('apply manually') || h.includes('apply directly') ||
                h.includes('apply on company site') || h.includes('apply on employer site'));
            if (!result) {
                // Priority 2: anything mentioning "apply" — unchanged from before, this is
                // exactly what matches Dice's "Easy Apply" button and every other site's
                // "Apply"/"Apply Now", so it keeps matching first, same as always.
                result = tryClick(h => h.includes('apply'));
            }
            if (!result) {
                // Priority 3 (fallback only): some external ATS forms use different
                // first-screen wording instead of "apply" at all.
                result = tryClick(h => h.includes('continue') || h.includes('get started') || h.includes('start application'));
            }
            return result || {clicked: false};
        """

        if already_has_form:
            # Only a single, non-looping probe here — not the full 15s
            # retrying search below. The button is either already rendered
            # right now (the page has clearly finished loading enough to
            # show the other fields) or it genuinely isn't there.
            try:
                result = self.driver.execute_script(find_and_click_js) or {}
            except Exception:
                result = {}
            if not result.get("clicked"):
                print("  Form fields already visible on landing. Skipping initial apply click.")
                if hasattr(self, "current_job_id") and self.current_job_id:
                    log_step(self.current_job_id, "Dice Agent", "  Form fields already visible on landing. Skipping initial apply click.")
                return True
            print(f"  [OK] Fields were visible but a distinct Apply button was also found — clicked: "
                  f"'{result.get('text', '')}' (href='{result.get('href', '')}')")
            if hasattr(self, "current_job_id") and self.current_job_id:
                log_step(self.current_job_id, "Dice Agent", f"  [OK] Fields were visible but a distinct Apply button was "
                          f"also found — clicked: '{result.get('text', '')}' (href='{result.get('href', '')}')")
            self._initial_apply_click_happened = True
            time.sleep(3.5)
            self.wait_for_cloudflare()
            return True

        print("  Form fields not visible. Searching for initial 'Apply' / 'Apply Now' button on portal...")

        if hasattr(self, "current_job_id") and self.current_job_id:

            log_step(self.current_job_id, "Dice Agent", "  Form fields not visible. Searching for initial 'Apply' / 'Apply Now' button on portal...")

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

                    log_step(self.current_job_id, "Dice Agent", f"  [DEBUG] Apply-button search script error: {e}")
                result = {}

            if result.get("clicked"):
                print(f"  [OK] Clicked initial apply button: '{result.get('text', '')}' (href='{result.get('href', '')}')")

                if hasattr(self, "current_job_id") and self.current_job_id:

                    log_step(self.current_job_id, "Dice Agent", f"  [OK] Clicked initial apply button: '{result.get('text', '')}' (href='{result.get('href', '')}')")
                self._initial_apply_click_happened = True
                time.sleep(3.5)  # Wait for page or form container to render
                self.wait_for_cloudflare()
                return True

            time.sleep(0.4)

        print("  [WARN] No initial apply button found or clicked.")


        if hasattr(self, "current_job_id") and self.current_job_id:


            log_step(self.current_job_id, "Dice Agent", "  [WARN] No initial apply button found or clicked.")
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

                                log_step(self.current_job_id, "Dice Agent", f"  [OK] State dropdown selected: {state_token}")
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

                                log_step(self.current_job_id, "Dice Agent", "  [OK] Country dropdown selected: United States")
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

                            log_step(self.current_job_id, "Dice Agent", f"  [OK] Clicked consent checkbox: {sel}")
        except Exception as ce:
            print(f"  [WARN] Checkbox clicking error: {ce}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Dice Agent", f"  [WARN] Checkbox clicking error: {ce}")

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

                                log_step(self.current_job_id, "Dice Agent", f"  [OK] Sponsorship dropdown auto-filled: {opt.text}")
                            break
                elif is_auth:
                    for opt in sel_obj.options:
                        opt_text = (opt.text or "").lower()
                        if opt_text.strip() in ("yes", "true", "y"):
                            sel_obj.select_by_visible_text(opt.text)
                            print(f"  [OK] Work Auth dropdown auto-filled: {opt.text}")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Dice Agent", f"  [OK] Work Auth dropdown auto-filled: {opt.text}")
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

                                log_step(self.current_job_id, "Dice Agent", f"  [OK] Relocation dropdown auto-filled: {opt.text}")
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

                                log_step(self.current_job_id, "Dice Agent", f"  [OK] Gender/self-ID dropdown auto-filled: {opt.text}")
                            break
        except Exception as se:
            print(f"  [WARN] Dropdown auto-filling error: {se}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Dice Agent", f"  [WARN] Dropdown auto-filling error: {se}")

    def _radio_label(self, el):
        gid = el.get_attribute("id") or ""
        label = ""
        if gid:
            try:
                label = self.driver.find_element(By.CSS_SELECTOR, f"label[for='{gid}']").text.strip()
            except Exception:
                pass
        return (label or el.get_attribute("value") or "").strip().lower()

    # Groups radio buttons that have neither a `name` nor an `id` by their shared
    # DOM container instead (confirmed live on a Ladders job reached via an
    # external redirect) — without this, such a group is completely invisible to
    # every other method in this class (all of them key off `name`).
    _UNNAMED_RADIO_GROUP_JS = """
        const GROUP_ATTR = 'data-bot-radio-group';
        function labelFor(radio) {
            const wrapper = radio.parentElement;
            if (wrapper) {
                const lbl = wrapper.querySelector('label');
                if (lbl && lbl.textContent.trim()) return lbl.textContent.trim();
            }
            return radio.getAttribute('aria-label') || radio.value || '';
        }
        function questionTextFor(container) {
            const parent = container.parentElement;
            if (!parent) return '';
            for (const child of parent.children) {
                if (child !== container) {
                    const text = (child.innerText || '').trim();
                    if (text && text.length < 200) return text;
                }
            }
            return '';
        }
        const radios = Array.from(document.querySelectorAll("input[type='radio']")).filter(r => {
            const rect = r.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0 && !r.disabled && !r.getAttribute('name');
        });
        let counter = window.__botRadioGroupCounter || 0;
        const containers = new Set();
        for (const r of radios) {
            let node = r.parentElement, container = null;
            for (let i = 0; i < 6 && node; i++) {
                if (node.querySelectorAll("input[type='radio']").length > 1) { container = node; break; }
                node = node.parentElement;
            }
            if (!container) continue;
            if (!container.getAttribute(GROUP_ATTR)) {
                counter += 1;
                container.setAttribute(GROUP_ATTR, 'g' + counter);
            }
            containers.add(container);
        }
        window.__botRadioGroupCounter = counter;

        const groups = [];
        containers.forEach(container => {
            const key = container.getAttribute(GROUP_ATTR);
            const opts = [];
            let anyChecked = false;
            container.querySelectorAll("input[type='radio']").forEach(r => {
                const label = labelFor(r);
                if (label) opts.push(label);
                if (r.checked) anyChecked = true;
            });
            if (opts.length && !anyChecked) {
                groups.push({key: key, question: questionTextFor(container), options: opts});
            }
        });
        return groups;
    """

    def _scan_unnamed_radio_groups(self):
        """Returns unfilled radio groups that have no name/id — see _UNNAMED_RADIO_GROUP_JS."""
        try:
            return self.driver.execute_script(self._UNNAMED_RADIO_GROUP_JS) or []
        except Exception as e:
            print(f"  [WARN] Unnamed radio-group scan error: {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Dice Agent", f"  [WARN] Unnamed radio-group scan error: {e}")
            return []

    def _stamp_synthetic_id(self, el, kind_tag):
        """
        Stamps a stable `data-bot-id` attribute on an element that has neither a
        real id nor a name, and returns a `__noid_<tag>_<n>` marker string that
        _apply_llm_field_answers() recognizes and resolves back via that attribute.
        """
        try:
            self._synthetic_id_counter = getattr(self, "_synthetic_id_counter", 0) + 1
            marker = f"s{self._synthetic_id_counter}"
            self.driver.execute_script("arguments[0].setAttribute('data-bot-id', arguments[1]);", el, marker)
            return f"__noid_{kind_tag}_{marker}"
        except Exception:
            return ""

    def _click_unnamed_radio_option(self, group_key, option_text):
        try:
            container = self.driver.find_element(By.CSS_SELECTOR, f"[data-bot-radio-group='{group_key}']")
            for r in container.find_elements(By.CSS_SELECTOR, "input[type='radio']"):
                if self._radio_label(r) == str(option_text).strip().lower() and not r.is_selected():
                    self.driver.execute_script("arguments[0].click();", r)
                    return True
        except Exception as e:
            print(f"  [WARN] Error clicking unnamed radio group '{group_key}': {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Dice Agent", f"  [WARN] Error clicking unnamed radio group '{group_key}': {e}")
        return False

    def _read_custom_widget_options(self, el, role):
        """
        Best-effort peek at a custom widget's option list. Opens it (JS click),
        reads visible [role='option']/li/.option text, then closes it again via
        Escape so page state is restored.
        """
        try:
            if role == "radiogroup":
                opts = []
                for child in el.find_elements(By.CSS_SELECTOR, "[role='radio']"):
                    text = (child.text or child.get_attribute("aria-label") or "").strip()
                    if text:
                        opts.append(text)
                return opts
            if role == "switch":
                return ["on", "off"]

            self.driver.execute_script("arguments[0].click();", el)
            time.sleep(0.35)
            opts = []
            for opt_el in self.driver.find_elements(By.CSS_SELECTOR, "[role='option'], [role='listbox'] li, [role='listbox'] div"):
                if opt_el.is_displayed():
                    text = (opt_el.text or "").strip()
                    if text and text not in opts:
                        opts.append(text)
                if len(opts) >= 12:
                    break
            self.driver.execute_script("document.activeElement && document.activeElement.blur();")
            try:
                el.send_keys(Keys.ESCAPE)
            except Exception:
                pass
            return opts
        except Exception:
            return []

    def _is_custom_widget_unfilled(self, el, role):
        try:
            if role == "radiogroup":
                children = el.find_elements(By.CSS_SELECTOR, "[role='radio']")
                return not any((c.get_attribute("aria-checked") or "").lower() == "true" for c in children)
            if role == "switch":
                return (el.get_attribute("aria-checked") or "").lower() not in ("true", "false")
            text = (el.text or el.get_attribute("aria-label") or "").strip().lower()
            placeholder_like = ("select" in text or "choose" in text or not text)
            checked_val = (el.get_attribute("aria-activedescendant") or "").strip()
            return placeholder_like and not checked_val
        except Exception:
            return False

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

                                log_step(self.current_job_id, "Dice Agent", f"  [OK] Relocation radio auto-filled ('{name}'): No")
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

                                log_step(self.current_job_id, "Dice Agent", f"  [OK] Gender/self-ID radio auto-filled ('{name}'): {label}")
                            break
        except Exception as e:
            print(f"  [WARN] Known radio-group auto-fill error: {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Dice Agent", f"  [WARN] Known radio-group auto-fill error: {e}")

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

                    log_step(self.current_job_id, "Dice Agent", f"  [OK] Date field filled with today's date: {today_iso}")
        except Exception as e:
            print(f"  [WARN] Date field auto-fill error: {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Dice Agent", f"  [WARN] Date field auto-fill error: {e}")

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

                    log_step(self.current_job_id, "Dice Agent", f"  [OK] Range slider set to: {default_percent}")
        except Exception as e:
            print(f"  [WARN] Range slider auto-fill error: {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Dice Agent", f"  [WARN] Range slider auto-fill error: {e}")

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

    # Anti-bot "honeypot" fields are deliberately kept in the DOM (often
    # visually hidden via off-screen CSS rather than display:none, so a plain
    # is_displayed() check doesn't catch them) — a real applicant would never
    # see or fill one, but a bot that treats every "unfilled" field as
    # answerable does. Confirmed live via the same external-apply pattern on a
    # FOX Careers / Workday signup form: a field literally labeled "Enter
    # website. This input is for robots only, do not enter if you're human."
    # got answered by the LLM like any other text field, which is the most
    # likely reason the site's "Create Account" click afterward had no
    # visible effect — a classic silent honeypot-triggered rejection, no
    # error shown. Mirrors the same fix in linkedin_apply.py.
    _HONEYPOT_TEXT_MARKERS = (
        "for robots only", "robots only", "if you're human", "if you are human",
        "leave this field blank", "leave blank", "do not fill", "do not enter",
        "should remain blank", "should be left blank", "honeypot",
    )

    def _is_honeypot_field(self, el, label_text=""):
        try:
            haystack = " ".join([
                label_text or "",
                el.get_attribute("name") or "",
                el.get_attribute("id") or "",
                el.get_attribute("placeholder") or "",
                el.get_attribute("aria-label") or "",
            ]).lower()
        except Exception:
            return False
        return any(marker in haystack for marker in self._HONEYPOT_TEXT_MARKERS)

    def _collect_unfilled_fields(self):
        """
        Scan the whole page for visible fields still needing a value/selection —
        including arbitrary <select> dropdowns, radio groups, and checkboxes beyond
        the deterministic identity/consent/sponsorship ones already handled — so the
        LLM pass below can answer them intelligently from the CV.
        """
        fields = []
        seen_radio_groups = set()
        self._custom_widget_map = {}
        seen_custom = set()
        try:
            # Password fields are deliberately excluded — a signup/account-creation
            # password must never be sent to the LLM as an "answerable" field. See
            # _has_unfilled_password_field() / _wait_for_password_entry().
            elements = self.driver.find_elements(
                By.CSS_SELECTOR,
                "input:not([type='hidden']):not([type='file']):not([type='password']), select, textarea"
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
                        select_id = eid
                        if not eid and not name:
                            # No id/name at all (e.g. a Ladders-style Gender/Race select
                            # reached via a Dice external redirect) — stamp one
                            # synthetically so an LLM answer can be applied back to it.
                            select_id = self._stamp_synthetic_id(el, "select")
                        fields.append({"kind": "select", "label": self._field_label(el), "name": name, "id": select_id, "options": options})
                        continue

                    current = (el.get_attribute("value") or "").strip()
                    if current:
                        continue
                    label_text = self._field_label(el)
                    if not label_text:
                        continue
                    if self._is_honeypot_field(el, label_text):
                        continue
                    fields.append({"kind": "text", "label": label_text, "name": name, "id": eid, "options": None})
                except Exception:
                    pass

            # Custom ARIA widgets — not reachable by the scan above at all.
            for el in self.driver.find_elements(By.CSS_SELECTOR, CUSTOM_WIDGET_ROLE_SELECTOR):
                try:
                    if not el.is_displayed():
                        continue
                    # NOTE: aria-haspopup="listbox" marks a dropdown that opens
                    # a listbox popup — it must fall through to the "combobox"
                    # default below, not be misread as a toggle switch. See
                    # matching fix/comment in linkedin_apply.py.
                    role = (el.get_attribute("role") or "").lower()
                    if not role:
                        role = "combobox"
                    dom_key = self.driver.execute_script(
                        "return arguments[0].outerHTML.slice(0,80) + '|' + Array.prototype.indexOf.call(document.querySelectorAll(arguments[1]), arguments[0]);",
                        el, CUSTOM_WIDGET_ROLE_SELECTOR,
                    )
                    if dom_key in seen_custom:
                        continue
                    seen_custom.add(dom_key)

                    if not self._is_custom_widget_unfilled(el, role):
                        continue

                    options = self._read_custom_widget_options(el, role)
                    if not options:
                        continue

                    # Stamp a stable attribute instead of holding onto this WebElement
                    # reference — these can go stale by the time an LLM answer comes
                    # back (the LLM round-trip takes real time, during which the
                    # site's own JS can re-render the widget). Re-querying fresh via
                    # this marker in _apply_llm_field_answers avoids that entirely.
                    marker = f"cw{len(self._custom_widget_map)}"
                    self.driver.execute_script("arguments[0].setAttribute('data-bot-custom-id', arguments[1]);", el, marker)
                    synthetic_id = f"__custom_{marker}"
                    self._custom_widget_map[synthetic_id] = marker
                    kind = "custom_radio" if role == "radiogroup" else ("custom_toggle" if role == "switch" else "custom_select")
                    fields.append({
                        "kind": kind,
                        "label": self._nearby_container_text(el)[:120] or self._field_label(el),
                        "name": "", "id": synthetic_id, "options": options,
                    })
                except Exception:
                    continue

            # Radio groups with no name/id, grouped by shared container (see
            # _scan_unnamed_radio_groups).
            for group in self._scan_unnamed_radio_groups():
                key = group.get("key")
                options = group.get("options") or []
                if not key or not options:
                    continue
                fields.append({
                    "kind": "radio",
                    "label": group.get("question") or "Question",
                    "name": "", "id": f"__radiogroup_{key}", "options": options,
                })
        except Exception:
            pass
        return fields

    def _llm_answer_fields(self, cv_data, fields, page_text):
        """Ask Groq to answer remaining unfilled fields (dropdowns, radios, open text) from the CV."""
        if not GROQ_API_KEY or not fields:
            return []
        try:
            llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0)
            fallback_llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0)
            prompt = f"""You are a job application assistant filling out an external company application form
(not LinkedIn/Indeed's own flow — this is a company careers site or ATS like Greenhouse/Workday/Lensa).

CURRENT PAGE TEXT (for context):
{page_text[:1200]}

VISIBLE UNFILLED FIELDS (kinds "select"/"radio"/"custom_select"/"custom_radio"/"custom_toggle"
are all pick-one-from-options fields — treat them identically):
{json.dumps(fields, separators=(',', ':'))}

CANDIDATE CV:
{json.dumps(_lean_cv_for_prompt(cv_data), separators=(',', ':'))}

TASK: Return the correct value for each field based ONLY on the candidate's CV.
Rules:
- For "select"/"radio"/"custom_select"/"custom_radio" fields, you MUST return one of the exact
  strings listed in that field's "options".
- For "custom_toggle" fields, return exactly "on" or "off".
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

                log_step(self.current_job_id, "Dice Agent", f"  [WARN] LLM form-answer error: {e}")
            return []

    def _apply_llm_field_answers(self, answers):
        for item in answers:
            fid = (item.get("id") or "").strip()
            fname = (item.get("name") or "").strip()
            val = item.get("value", "")
            if not val or (not fid and not fname):
                continue
            try:
                # No-name/no-id radio group (see _scan_unnamed_radio_groups) — click
                # the option within the stamped container whose label matches.
                if fid.startswith("__radiogroup_"):
                    group_key = fid[len("__radiogroup_"):]
                    if self._click_unnamed_radio_option(group_key, val):
                        print(f"  [OK] LLM-filled radio group '{group_key}' -> '{val}'")

                        if hasattr(self, "current_job_id") and self.current_job_id:

                            log_step(self.current_job_id, "Dice Agent", f"  [OK] LLM-filled radio group '{group_key}' -> '{val}'")
                    else:
                        print(f"  [WARN] LLM answer '{val}' had no matching option in radio group '{group_key}'")

                        if hasattr(self, "current_job_id") and self.current_job_id:

                            log_step(self.current_job_id, "Dice Agent", f"  [WARN] LLM answer '{val}' had no matching option in radio group '{group_key}'")
                    continue

                # Custom ARIA widget — re-query fresh via the stamped marker rather
                # than trusting a WebElement reference held since scan time, which
                # can go stale by the time this runs.
                if fid.startswith("__custom_") and fid in self._custom_widget_map:
                    marker = self._custom_widget_map[fid]
                    matches = self.driver.find_elements(By.CSS_SELECTOR, f"[data-bot-custom-id='{marker}']")
                    if not matches:
                        print(f"  [WARN] Custom widget '{fid}' no longer present on the page — skipping.")

                        if hasattr(self, "current_job_id") and self.current_job_id:

                            log_step(self.current_job_id, "Dice Agent", f"  [WARN] Custom widget '{fid}' no longer present on the page — skipping.")
                        continue
                    el = matches[0]
                    role = (el.get_attribute("role") or "").lower()
                    if role == "radiogroup":
                        for child in el.find_elements(By.CSS_SELECTOR, "[role='radio']"):
                            label = (child.text or child.get_attribute("aria-label") or "").strip()
                            if label.lower() == str(val).strip().lower():
                                self.driver.execute_script("arguments[0].click();", child)
                                print(f"  [OK] LLM-filled custom radio '{fid}' -> '{val}'")

                                if hasattr(self, "current_job_id") and self.current_job_id:

                                    log_step(self.current_job_id, "Dice Agent", f"  [OK] LLM-filled custom radio '{fid}' -> '{val}'")
                                break
                    elif role == "switch":
                        want_on = str(val).strip().lower() == "on"
                        is_on = (el.get_attribute("aria-checked") or "").lower() == "true"
                        if want_on != is_on:
                            self.driver.execute_script("arguments[0].click();", el)
                            print(f"  [OK] LLM-filled custom toggle '{fid}' -> '{val}'")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Dice Agent", f"  [OK] LLM-filled custom toggle '{fid}' -> '{val}'")
                    else:
                        self.driver.execute_script("arguments[0].click();", el)
                        time.sleep(0.35)
                        matched = False
                        for opt_el in self.driver.find_elements(By.CSS_SELECTOR, "[role='option']"):
                            if opt_el.is_displayed() and (opt_el.text or "").strip().lower() == str(val).strip().lower():
                                self.driver.execute_script("arguments[0].click();", opt_el)
                                matched = True
                                break
                        if matched:
                            print(f"  [OK] LLM-filled custom dropdown '{fid}' -> '{val}'")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Dice Agent", f"  [OK] LLM-filled custom dropdown '{fid}' -> '{val}'")
                        else:
                            try:
                                el.send_keys(Keys.ESCAPE)
                            except Exception:
                                pass
                            print(f"  [WARN] LLM answer '{val}' had no matching option in custom dropdown '{fid}'")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Dice Agent", f"  [WARN] LLM answer '{val}' had no matching option in custom dropdown '{fid}'")
                    continue

                el = None
                if fid.startswith("__noid_"):
                    # Synthetic marker from _stamp_synthetic_id() — resolve via the
                    # data-bot-id attribute stamped on it during scanning, since this
                    # element genuinely has no real id/name to match on at all.
                    marker = fid.rsplit("_", 1)[-1]
                    matches = self.driver.find_elements(By.CSS_SELECTOR, f"[data-bot-id='{marker}']")
                    el = matches[0] if matches else None
                elif fid:
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

                                log_step(self.current_job_id, "Dice Agent", f"  [OK] LLM-filled dropdown '{fid or fname}' -> '{val}'")
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

                                log_step(self.current_job_id, "Dice Agent", f"  [OK] LLM-checked checkbox '{fid or fname}'")
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

                                log_step(self.current_job_id, "Dice Agent", f"  [OK] LLM-filled field '{fid or fname}' -> '{str(val)[:40]}'")
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

                                log_step(self.current_job_id, "Dice Agent", f"  [OK] LLM-selected radio '{fname}' -> '{val}'")
                            break
            except Exception:
                pass

    def _page_signature(self):
        try:
            return self.driver.find_element(By.TAG_NAME, "body").text[:3000]
        except Exception:
            return ""

    # Final-step button text is worded very differently across ATS platforms
    # (confirmed live: Ladders uses "Apply to Job", not "Submit") — this list is
    # intentionally broad since under-matching here is the dangerous direction: a
    # final button that isn't recognized falls through to _click_continue_button(),
    # which could click it (and truly submit the application) if it also happens
    # to be a type="submit" element, instead of correctly stopping for human review.
    _FINAL_STEP_KEYWORDS = (
        "submit", "send application", "apply now", "apply to job", "apply for this job",
        "finish application", "complete application", "complete my application",
    )

    # Some ATS portals insert a bare Terms-of-Use / consent interstitial before
    # the real application form — confirmed live on FOX Careers (fronts a
    # Workday application) via the LinkedIn external-apply path: a page with NO
    # form fields at all, just legal boilerplate text and a single
    # '<input type="submit" id="AgreementButton" value="Submit">' button. The
    # old code treated that lone "Submit" button as _is_final_submit_step() ==
    # True and returned immediately — permanently stopping fill_form() before
    # the real form was ever reached. Mirrors the same fix applied in
    # linkedin_apply.py. Deliberately narrow: only fires when the page has
    # literally NO form-field content (_frame_has_form_content() False) AND the
    # visible text contains one of these unambiguous consent/ToU phrases — a
    # genuine "ready to submit" review page always either still shows its
    # filled-in fields or won't match these phrases, so this can never be
    # confused with an actual application submission.
    _AGREEMENT_GATE_TEXT_MARKERS = (
        "terms of use", "privacy statement", "privacy policy", "i acknowledge",
        "i agree to abide by", "i have read and agree",
    )

    def _is_bare_agreement_gate(self):
        if self._frame_has_form_content():
            return False
        try:
            text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
        except Exception:
            return False
        return any(marker in text for marker in self._AGREEMENT_GATE_TEXT_MARKERS)

    # Some ATS forms overlay their real, interactive button with a second
    # decoy/tracking element positioned exactly on top of it — confirmed live
    # via the same external-apply path on a FOX Careers / Workday "Create
    # Account" step: the real
    # <button data-automation-id="createAccountSubmitButton"> is
    # aria-hidden="true" and tabindex="-2" (i.e. genuinely non-interactive),
    # while a <div role="button" data-automation-id="click_filter"
    # aria-label="Create Account"> sits visually on top of it and is what the
    # site's own click handler is actually wired to. Selenium's real
    # el.click() correctly raised ElementClickIntercepted here — but the old
    # exception fallback (a blind execute_script("arguments[0].click()"))
    # bypassed that check entirely and force-clicked the hidden, non-
    # functional button underneath, which is why the page never advanced
    # despite every field being filled correctly and the click "succeeding".
    # This clicks whatever element is ACTUALLY at that point on screen
    # instead — the same thing a real mouse click would hit. Mirrors the
    # same fix in linkedin_apply.py.
    _CLICK_THROUGH_OVERLAY_JS = """
        const el = arguments[0];
        const rect = el.getBoundingClientRect();
        const x = rect.left + rect.width / 2;
        const y = rect.top + rect.height / 2;
        const top = document.elementFromPoint(x, y);
        (top || el).click();
    """

    def _click_resilient(self, el):
        """Real click first; on interception (or any other failure), clicks
        whatever element is actually on top at that point instead of blindly
        force-clicking the original (possibly hidden/decoy) target."""
        try:
            el.click()
        except Exception:
            try:
                self.driver.execute_script(self._CLICK_THROUGH_OVERLAY_JS, el)
            except Exception:
                self.driver.execute_script("arguments[0].click();", el)

    def _click_bare_agreement_gate(self):
        """If _is_bare_agreement_gate() is True, clicks the lone Agree/Accept/
        Acknowledge button so the flow can proceed to the real form. Returns
        False (no-op) on any page that has real form content, or that doesn't
        match the consent-page text markers — never touches an actual submit
        step.

        "submit" is deliberately excluded from the clickable-text list below
        — confirmed live on two real Workday jobs: a genuine final
        review/submit page can also have _frame_has_form_content() == False
        (previously-entered data shown as read-only summary text) AND
        boilerplate text mentioning "Privacy Policy"/"Terms of Use" near the
        Submit button (an almost universal pattern on final application
        pages), which together satisfied _is_bare_agreement_gate()'s trigger
        — so this was auto-clicking the REAL application Submit button.
        Mirrors linkedin_apply.py.
        """
        if not self._is_bare_agreement_gate():
            return False
        selectors = ["input[type='submit']", "input[type='button']", "xpath://button"]
        for sel in selectors:
            by_type = By.XPATH if sel.startswith("xpath:") else By.CSS_SELECTOR
            actual_sel = sel[6:] if sel.startswith("xpath:") else sel
            try:
                for el in self.driver.find_elements(by_type, actual_sel):
                    if not (el.is_displayed() and el.is_enabled()):
                        continue
                    text = (el.text or el.get_attribute("value") or "").strip().lower()
                    if "submit" in text:
                        continue
                    if any(k in text for k in ("agree", "accept", "acknowledge", "continue", "next")):
                        self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                        time.sleep(0.2)
                        self._click_resilient(el)
                        print(f"  [OK] Clicked through an intermediate Terms-of-Use/agreement gate: '{text}'")
                        return True
            except Exception:
                pass
        return False

    def _is_final_submit_step(self):
        """True if a final Submit/Apply button (not Next/Continue) is visible on the current step."""
        try:
            selectors = [
                "xpath://button",
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
                        if any(k in text for k in self._FINAL_STEP_KEYWORDS):
                            return True
        except Exception:
            pass
        return False

    _CONTINUE_BUTTON_SELECTORS = [
        "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue')]",
        "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'next')]",
        "xpath://a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue')]",
        "button[type='submit']",
    ]

    def _has_disabled_continue_button(self):
        """
        True only if a Continue/Next/Save-and-Continue-style button actually
        EXISTS and is visible but currently disabled — a specific signal that
        a required field is blocking progress (confirmed live on Elevance
        Health / Workday: "Save and Continue" stayed disabled because a
        custom dropdown and Yes/No question weren't recognized as fillable).
        Deliberately narrower than "no button was clickable" — a page that
        has genuinely finished and has no such button at all must NOT
        trigger this, or it reintroduces the indefinite-hang bug fixed
        earlier. Mirrors linkedin_apply.py.
        """
        for sel in self._CONTINUE_BUTTON_SELECTORS:
            try:
                by_type = By.XPATH if sel.startswith("xpath:") else By.CSS_SELECTOR
                actual_sel = sel[6:] if sel.startswith("xpath:") else sel
                for el in self.driver.find_elements(by_type, actual_sel):
                    if el.is_displayed() and not el.is_enabled():
                        return True
            except Exception:
                pass
        return False

    def _has_required_empty_file_field(self):
        """
        True if a visible, required file-upload field (Resume/CV, etc.) is
        still empty — a signal that the step is genuinely NOT finished yet,
        even when no Continue/Submit button exists at all. Confirmed live on
        a New York Power Authority job reached via LinkedIn's external-apply
        path (same underlying pattern applies here): the page had no button
        matching any Continue-button selector, so the old logic fell
        straight through to "no further button found, treating as final"
        and silently abandoned the job with a required Resume field still
        empty. Deliberately narrow — only fires when a field is both
        unfilled (no data-bot-uploaded marker, i.e. _try_upload_file() never
        succeeded on it) AND flagged required (native `required` attribute,
        or a nearby "*" marker next to resume/cv/cover text) — never on a
        form that's genuinely finished. Mirrors linkedin_apply.py.
        """
        try:
            for el in self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']"):
                if not el.is_displayed() or el.get_attribute("data-bot-uploaded"):
                    continue
                if el.get_attribute("required"):
                    return True
                context = self._nearby_container_text(el)
                if "*" in context and any(k in context for k in ("resume", "cv", "cover")):
                    return True
        except Exception:
            pass
        return False

    def _click_continue_button(self):
        """Click a Next/Continue/Save & Continue button to advance a multi-step form. Never clicks a Submit button."""
        for sel in self._CONTINUE_BUTTON_SELECTORS:
            try:
                by_type = By.XPATH if sel.startswith("xpath:") else By.CSS_SELECTOR
                actual_sel = sel[6:] if sel.startswith("xpath:") else sel
                elements = self.driver.find_elements(by_type, actual_sel)
                for el in elements:
                    if not (el.is_displayed() and el.is_enabled()):
                        continue
                    text = (el.text or el.get_attribute("value") or "").strip().lower()
                    # Absolute, unconditional exclusion — never click anything
                    # whose visible text contains "submit", no matter what else
                    # it says. The "continue"/"next" carve-out just below this
                    # exists only so a legitimate wizard-navigation button that
                    # happens to be HTML button[type='submit'] (a technicality —
                    # its text is just "Continue", nothing "submit"-related)
                    # doesn't get wrongly excluded. It must never also excuse a
                    # button whose actual wording contains "submit" (e.g. a
                    # composite "Continue and Submit" / "Review and Submit"
                    # label) — that combination used to slip through here and
                    # actually submit the real application, confirmed live on
                    # two Workday jobs where the bot filled the form correctly
                    # and then went on to press the real Submit button itself.
                    # This must only ever be clicked by a human, after review.
                    # Mirrors linkedin_apply.py.
                    if "submit" in text:
                        continue
                    if "continue" not in text and "next" not in text and \
                            any(k in text for k in self._FINAL_STEP_KEYWORDS):
                        continue
                    # Confirmed live on a Randstad Digital / HTC Global Services job
                    # (both route through the same Ceipal-style ATS): a "Login" button
                    # has type='submit', so it was being caught by the catch-all above
                    # and repeatedly clicked as if it were a normal "continue" step —
                    # it can't actually be completed by the bot (needs the candidate's
                    # real credentials / email verification) and the page never
                    # advances, so this just wasted retries before giving up. An
                    # auth-gate button is never a legitimate "advance the form" click.
                    if any(k in text for k in ("log in", "login", "sign in", "sign-in")):
                        continue
                    self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                    time.sleep(0.2)
                    self._click_resilient(el)
                    print(f"  [OK] Clicked continue/next: '{text}'")

                    if hasattr(self, "current_job_id") and self.current_job_id:

                        log_step(self.current_job_id, "Dice Agent", f"  [OK] Clicked continue/next: '{text}'")
                    return True
            except Exception:
                pass
        return False

    def _has_unfilled_password_field(self):
        """
        True if a visible <input type="password"> on the page still has no value —
        e.g. an external Dice redirect landing on a "create your account" signup
        form requiring a password the bot has no safe way to invent. Checks ALL
        password fields (a signup form commonly has both "Password" and "Confirm
        password"), not just the first.
        """
        try:
            for el in self.driver.find_elements(By.CSS_SELECTOR, "input[type='password']"):
                if el.is_displayed() and not (el.get_attribute("value") or "").strip():
                    return True
        except Exception:
            pass
        return False

    def _wait_for_password_entry(self):
        """
        Pauses and waits for you to type your password (and confirm-password, if
        present) into the signup form yourself — the bot never guesses or invents
        one. Polls until every visible password field has a value, refreshing the
        SAME review-deadline the dashboard already shows (not a second, separate
        timer) each cycle, so the countdown doesn't expire out from under you while
        you're mid-typing. No hard cutoff of its own: if the tab/browser is closed,
        the driver calls here raise and unwind normally through the existing error
        handling in fill_form(), same as everywhere else in this class.
        """
        print("  [WARN] Waiting for you to create your password — please enter and confirm it, "
              "the bot will continue automatically.")

        if hasattr(self, "current_job_id") and self.current_job_id:

            log_step(self.current_job_id, "Dice Agent", "  [WARN] Waiting for you to create your password — please enter and confirm it, "
                     "the bot will continue automatically.")

        while self._has_unfilled_password_field():
            try:
                from src.workflow.apply import skip_event, SkipRequested, check_browser_alive
                if skip_event.is_set():
                    raise SkipRequested()
                check_browser_alive(self.driver)
            except ImportError:
                pass
            time.sleep(2)

        print("  [OK] Password entered — continuing automatically.")

        if hasattr(self, "current_job_id") and self.current_job_id:

            log_step(self.current_job_id, "Dice Agent", "  [OK] Password entered — continuing automatically.")

    def _detect_validation_errors(self):
        """
        When a Continue click has no visible effect, scan for validation-error
        indicators instead of just giving up silently — so the log states exactly
        which field is blocking progress. Mirrors linkedin_apply.py.
        """
        found = []
        try:
            for sel in _ERROR_SELECTORS:
                for el in self.driver.find_elements(By.CSS_SELECTOR, sel):
                    if not el.is_displayed():
                        continue
                    text = (el.text or "").strip()
                    context = self._nearby_container_text(el)[:150]
                    if text or context:
                        found.append(text or context)
                    if len(found) >= 8:
                        break
        except Exception:
            pass
        seen = set()
        unique = []
        for f in found:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return unique

    def wait_for_page_ready(self, timeout=12):
        """
        Poll until at least one real interactive element (native or custom-widget)
        is visible on the page, instead of assuming the page is ready the instant
        its text changes. Mirrors linkedin_apply.py's wait_for_page_ready().
        """
        deadline = time.time() + timeout
        probe_selector = (
            "input:not([type='hidden']), select, textarea, button, "
            + CUSTOM_WIDGET_ROLE_SELECTOR
        )
        last_count = -1
        stable_reads = 0
        while time.time() < deadline:
            try:
                visible = [el for el in self.driver.find_elements(By.CSS_SELECTOR, probe_selector) if el.is_displayed()]
                count = len(visible)
            except Exception:
                count = 0
            if count > 0:
                if count == last_count:
                    stable_reads += 1
                    if stable_reads >= 2:
                        return True
                else:
                    stable_reads = 0
                last_count = count
            time.sleep(0.6)
        return last_count > 0

    # Injected directly into the page currently open in the automated browser
    # (not just the log/dashboard) — a fixed banner with a live countdown.
    # Mirrors linkedin_apply.py's _SHOW_STUCK_BANNER_JS.
    # pointer-events:none is required — without it, this full-width bar
    # pinned to the top of the viewport physically intercepts clicks on any
    # field/button that happens to sit near the top of the page underneath
    # it, blocking the exact manual fix it's asking you to make. Caught live
    # via a synthetic regression test (ElementClickIntercepted pointing at
    # this banner). Mirrors linkedin_apply.py.
    _SHOW_STUCK_BANNER_JS = """
        var msg = arguments[0];
        var seconds = arguments[1];
        var old = document.getElementById('__jobbot_stuck_banner');
        if (old) old.remove();
        var el = document.createElement('div');
        el.id = '__jobbot_stuck_banner';
        el.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:2147483647;' +
            'background:#c0392b;color:#fff;font:bold 16px/1.4 Arial,sans-serif;' +
            'padding:12px 16px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.4);' +
            'pointer-events:none;';
        el.innerText = msg + ' You have ' + seconds + ' seconds.';
        document.documentElement.appendChild(el);
        var remaining = seconds;
        var iv = setInterval(function() {
            remaining -= 1;
            var live = document.getElementById('__jobbot_stuck_banner');
            if (!live) { clearInterval(iv); return; }
            if (remaining > 0) {
                live.innerText = msg + ' You have ' + remaining + ' seconds.';
            } else {
                live.innerText = msg + ' Please complete this step now — the bot will resume automatically once you do.';
                clearInterval(iv);
            }
        }, 1000);
    """
    _HIDE_STUCK_BANNER_JS = "var el = document.getElementById('__jobbot_stuck_banner'); if (el) el.remove();"

    def _show_stuck_banner(self, message="Bot failed — please try manually.", seconds=15):
        try:
            self.driver.execute_script(self._SHOW_STUCK_BANNER_JS, message, seconds)
        except Exception:
            pass

    def _hide_stuck_banner(self):
        try:
            self.driver.execute_script(self._HIDE_STUCK_BANNER_JS)
        except Exception:
            pass

    def _wait_for_human_to_advance_step(self, page_signature_before, message="Bot failed — please try manually."):
        """
        Pauses when the bot's single fill-then-click attempt on this step
        didn't genuinely work — the page came back unchanged, or with
        validation errors — instead of looping back to blindly re-fill and
        re-click the same step again. Waits for you to finish whatever's
        left on THIS step and click Next/Continue/Submit yourself,
        refreshing the SAME 180-second review-deadline the dashboard
        already shows each cycle (not a second, separate timer) — same
        mechanism as _wait_for_password_entry() above.

        Requires BOTH the page to have changed AND no validation errors to
        still be visible before resuming — confirmed live via the same
        external-apply path on a Workday-based application (GM Motorsports)
        that just comparing raw page text was not reliable enough: Workday's
        own page re-renders something trivial (tooltip/focus state) within
        1-2 seconds on its own, with nobody touching the page, which looked
        like "the page changed" and made the bot resume immediately and
        re-attempt the exact same broken fields — over and over, every few
        seconds, never actually stopping to wait for a real human action.
        Checking that the specific validation errors are gone (not just that
        "some text" changed) is a much stronger signal that you genuinely
        completed the step. Mirrors the same fix in linkedin_apply.py.
        """
        try:
            handles_before_wait = self.driver.window_handles
        except Exception:
            handles_before_wait = []

        self._show_stuck_banner(message)
        while True:
            # Lets a Skip click take effect immediately even while stuck here
            # — previously nothing checked for it until fill_form() returned
            # and REVIEW REQUIRED was reached, so a job stuck in this exact
            # loop (confirmed live on QTech US Inc, repeatedly failing to
            # advance past its own Easy Apply wizard) had no way to be
            # skipped at all. Caught and handled in apply_to_job().
            try:
                from src.workflow.apply import skip_event, SkipRequested, check_browser_alive
                if skip_event.is_set():
                    raise SkipRequested()
                check_browser_alive(self.driver)
            except ImportError:
                pass

            # A brand-new tab/window can open for the real form WITHOUT the
            # interim tab ever closing — confirmed live on the same
            # TEKsystems c/o Allegis Group job, second failure mode: the old
            # tab stayed alive (so the dead-window check below never fires)
            # and its own body text never changed (so the signature check
            # below never fires either), while the real, fillable form sat
            # untouched in a sibling window the driver never switched into.
            # Checking for an extra handle FIRST, before either of the other
            # checks, catches this the same way the very first Apply click
            # already does at the top of fill_form().
            try:
                current_handles = self.driver.window_handles
            except Exception:
                current_handles = []
            new_handles = [h for h in current_handles if h not in handles_before_wait]
            if new_handles:
                try:
                    self.driver.switch_to.window(new_handles[-1])
                    print("  [OK] A new tab/window opened — switched to it to resume filling.")
                    if hasattr(self, "current_job_id") and self.current_job_id:
                        log_step(self.current_job_id, "Dice Agent", "  [OK] A new tab/window opened — switched to it to resume filling.")
                    break
                except Exception:
                    pass

            # _page_signature()'s own try/except returns "" when the current
            # tab is dead, which used to look identical to "the human changed
            # the page" and broke out of this loop pointed at a closed window
            # — confirmed live on the same TEKsystems c/o Allegis Group job,
            # first failure mode: the intermediate tab closed itself on its
            # own (same auto-redirect-then-close pattern already documented
            # in _recover_dead_window(), just triggered by the site's own
            # timing instead of a click), and every fill attempt afterward
            # failed with "invalid session id: target window already closed"
            # because the driver was still pointed at the dead handle.
            # Checking window liveness next, and hopping onto the new live
            # tab via the same _recover_dead_window() used elsewhere, before
            # ever trusting a "page changed" reading, fixes this without
            # weakening the genuine human-completed-the-step case.
            try:
                self.driver.current_window_handle
                window_alive = True
            except Exception:
                window_alive = False

            if not window_alive:
                if self._recover_dead_window():
                    break
                time.sleep(2)
                continue

            try:
                current_signature = self._page_signature()
            except Exception:
                current_signature = page_signature_before
            if current_signature != page_signature_before and not self._detect_validation_errors():
                break
            try:
                pass # Timer reset removed
            except Exception:
                pass
            time.sleep(2)

        self._hide_stuck_banner()
        # A page-signature change on a slow-hydrating ATS (Workday confirmed
        # live) can just mean a loading state replaced the previous text — not
        # that the real form has actually rendered yet. Waiting here for
        # genuine interactive elements before declaring "resumed" avoids
        # immediately running a fill pass against a still-empty page and
        # concluding there's nothing left to do. Mirrors linkedin_apply.py.
        self.wait_for_page_ready(timeout=12)
        self._just_resumed_from_human = True

        print("  [OK] Page changed and no validation errors remain — resuming automatic filling.")

        if hasattr(self, "current_job_id") and self.current_job_id:

            log_step(self.current_job_id, "Dice Agent", "  [OK] Page changed and no validation errors remain — resuming automatic filling.")

    def _generate_strong_password(self, length=16):
        """
        A cryptographically random password (via `secrets`, not `random`) that
        satisfies typical signup-form complexity rules (upper/lower/digit/symbol,
        length >= 16). Symbol set is deliberately narrow — avoids quote/backslash/
        backtick characters that could break an HTML attribute value or a site's
        own client-side format validation.
        """
        upper, lower, digits, symbols = string.ascii_uppercase, string.ascii_lowercase, string.digits, "!@#$%^&*-_=+"
        all_chars = upper + lower + digits + symbols
        while True:
            chars = [secrets.choice(upper), secrets.choice(lower), secrets.choice(digits), secrets.choice(symbols)]
            chars += [secrets.choice(all_chars) for _ in range(length - len(chars))]
            secrets.SystemRandom().shuffle(chars)
            password = "".join(chars)
            if (any(c in upper for c in password) and any(c in lower for c in password)
                    and any(c in digits for c in password) and any(c in symbols for c in password)):
                return password

    def _save_generated_password(self, password, email):
        """
        Writes the generated signup password to a durable per-job file (alongside
        the tailored CV/cover letter already saved there for this job) — without
        this, a password only the bot ever saw would lock you out of an account
        it just created in your name the moment the log scrolls past it.
        """
        try:
            job_dir = os.path.join(APPLICATIONS_DIR, self.current_job_id or "unknown-job")
            os.makedirs(job_dir, exist_ok=True)
            path = os.path.join(job_dir, "account_credentials.txt")
            try:
                site = self.driver.current_url
            except Exception:
                site = "unknown"
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}]\nSite: {site}\nEmail: {email}\nPassword: {password}\n\n")
            return path
        except Exception:
            return None

    def _fill_all_password_fields(self, password):
        """
        Fills the given password into every empty visible password field on
        the page (covers the "Password" + "Confirm password" pair on a
        signup form, or the single field on a sign-in form). Returns True
        only if at least one field was actually filled.
        """
        filled_any = False
        for el in self.driver.find_elements(By.CSS_SELECTOR, "input[type='password']"):
            try:
                if not (el.is_displayed() and not (el.get_attribute("value") or "").strip()):
                    continue
                try:
                    el.click()
                except Exception:
                    pass
                # Real per-character keystrokes first, matching the pattern already
                # proven reliable for other LLM-answered text fields — confirmed live
                # via the same external-apply path on a FOX Careers / Workday signup
                # form: the JS native-setter path used here previously left both
                # password fields showing the correct value in the DOM, yet the
                # "Create Account" button's own click still did nothing at all, even
                # after waiting — consistent with a React/Formik-style form whose
                # internal validity state didn't get updated because no real input
                # events were ever fired. Falls back to the JS setter only if
                # send_keys didn't actually take.
                try:
                    self.driver.execute_script(_NATIVE_SET_VALUE_JS, el, "")
                    el.send_keys(password)
                except Exception:
                    pass
                final_value = (el.get_attribute("value") or "").strip()
                if final_value != password:
                    try:
                        self.driver.execute_script(_NATIVE_SET_VALUE_JS, el, password)
                    except Exception:
                        pass
                try:
                    self.driver.execute_script("arguments[0].dispatchEvent(new Event('blur', {bubbles: true}));", el)
                except Exception:
                    pass
                filled_any = True
            except Exception:
                continue
        return filled_any

    def _generate_and_fill_password(self):
        """
        Generates one strong password and fills it into every empty visible
        password field on the page (covers the "Password" + "Confirm password"
        pair) so a signup wall no longer needs you to stop and type one by hand.
        Saves the generated password via _save_generated_password() first —
        before touching the page — so it's on disk even if the fill itself fails
        partway through. Returns True only if at least one field was actually
        filled; the caller falls back to _wait_for_password_entry() otherwise.
        """
        password = self._generate_strong_password()
        email = self.cv_data.get("email", "") or CANDIDATE_EMAIL
        saved_path = self._save_generated_password(password, email)
        if saved_path:
            print(f"  [WARN] Generated a strong signup password and saved it to {saved_path}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Dice Agent", f"  [WARN] Generated a strong signup password and saved it to {saved_path}")
        else:
            print(f"  [WARN] Generated a strong signup password but could not save it to a file — check this line for it: {password}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Dice Agent", f"  [WARN] Generated a strong signup password but could not save it to a file — check this line for it: {password}")

        filled_any = self._fill_all_password_fields(password)
        if filled_any:
            print("  [OK] Filled the generated password into the signup form.")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Dice Agent", "  [OK] Filled the generated password into the signup form.")
        return filled_any

    def _read_saved_password(self):
        """
        Reads back the most recently generated password for THIS job from
        its saved credentials file — used when the bot lands on a Sign-In
        gate (not a fresh signup) so it doesn't generate and fill a brand
        new, wrong password that could never match the account that was
        actually created earlier in this same run. Mirrors linkedin_apply.py.
        """
        try:
            path = os.path.join(APPLICATIONS_DIR, self.current_job_id or "unknown-job", "account_credentials.txt")
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            matches = re.findall(r"^Password:\s*(.+)$", content, re.MULTILINE)
            return matches[-1].strip() if matches else None
        except Exception:
            return None

    def _has_auth_gate_button(self):
        """True if a visible Login/Sign-in button/link is on the page — used to tell
        a genuine "nothing left to do" final step apart from a portal that's actually
        gating the real form behind credentials the bot doesn't have."""
        try:
            for el in self.driver.find_elements(By.CSS_SELECTOR, "button, a, input[type='submit']"):
                if not (el.is_displayed() and el.is_enabled()):
                    continue
                text = (el.text or el.get_attribute("value") or "").strip().lower()
                if any(k in text for k in ("log in", "login", "sign in", "sign-in")):
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
            real_click_happened = self._initial_apply_click_happened

            if clicked:
                print("  [Dice Agent] Checking for external redirect...")
                if hasattr(self, "current_job_id") and self.current_job_id:
                    log_step(self.current_job_id, "Dice Agent", "Checking for external redirect...")

                time.sleep(3.5)
                handles_after_click = self.driver.window_handles
                
                if len(handles_after_click) > len(handles_before_click):
                    new_tab = [h for h in handles_after_click if h not in handles_before_click][-1]
                    self.driver.switch_to.window(new_tab)
                    print("  Switched to new tab/window for external Dice application.")
                    if hasattr(self, "current_job_id") and self.current_job_id:
                        log_step(self.current_job_id, "Dice Agent", "Switched to new tab/window for external Dice application.")
                else:
                    print("  No new tab detected, continuing application in current tab.")
                    if hasattr(self, "current_job_id") and self.current_job_id:
                        log_step(self.current_job_id, "Dice Agent", "No new tab detected, continuing application in current tab.")

            # Safety net for the same tab-closes-itself pattern _recover_dead_window()
            # handles below — cheap no-op if the current tab is still fine.
            self._recover_dead_window()

            self.wait_for_cloudflare()

            # The page reached after switching tabs isn't always the real form either —
            # confirmed live on a Randstad Digital job: Dice's own "Apply" click opened a
            # new tab showing the job re-posted on Randstad's own careers site
            # (randstadusa.com), which itself still needed its own "Apply" button clicked
            # before the real form ever appeared. _click_initial_apply_button() already
            # no-ops safely if the real form is already visible (via _frame_has_form_content),
            # so calling it again here is safe on every job that doesn't hit this pattern.
            if clicked:
                self._click_initial_apply_button()
                real_click_happened = real_click_happened or self._initial_apply_click_happened
                self.wait_for_cloudflare()

            # Some ATS forms reached via a Dice external redirect are embedded in an
            # <iframe> on the company's own page rather than served directly — this
            # is a no-op if the top-level page already has real form content
            # (including Dice's own wizard, which is never iframe-embedded).
            self._switch_into_form_iframe_if_needed()

            # Only verify/escalate when a click actually happened (something
            # matched "apply" and got clicked) but didn't lead anywhere real —
            # NOT when nothing was found to click at all. A page with
            # genuinely no apply button and no form (confirmed live on a
            # MANTECH job used in regression testing) must keep falling
            # through to the existing "no button found" handling unchanged;
            # escalating on that case too would wait forever for a human to
            # click a button that was never there to begin with. Deliberately
            # keyed off whether a REAL click happened (not off whether the
            # page already looked like it had a form before the click) —
            # confirmed live on a SIERTEK LTD / LinkedIn-external-apply job:
            # form-like fields were already visible (an unrelated quick-apply/
            # search widget), yet the bot still went on to click a distinct
            # real Apply button, and that click needs the exact same
            # verification as the "no form at all yet" case, not to be
            # silently skipped. Mirrors linkedin_apply.py.
            if real_click_happened:
                # Verify the initial apply click(s) actually led to a real
                # application form before trusting it and filling anything —
                # confirmed live on Yamaha Motor Corporation (via the
                # LinkedIn-external-apply path, same underlying pattern
                # applies here): the generic "apply"-text matcher can grab a
                # site's own top-nav "APPLY NOW" menu item instead of the
                # real per-job Apply link, revealing the site's own
                # job-SEARCH bar — which then gets filled in as if it were
                # the real application. Rather than trying to out-guess every
                # possible site's DOM structure, this tries automatically
                # exactly once and, if that didn't lead to a genuine form,
                # asks you to find and click the real Apply button yourself —
                # the bot resumes automatically once a real form appears,
                # same as every other wait-for-human step. Mirrors
                # linkedin_apply.py.
                landed_on_real_form = False
                try:
                    landed_on_real_form = self._frame_has_form_content()
                except Exception:
                    pass
                if not landed_on_real_form:
                    print("  [WARN] This step needs your help — couldn't confirm a real application "
                          "form opened. Please find and click the Apply button yourself; "
                          "the bot will resume automatically once the form appears.")

                    if hasattr(self, "current_job_id") and self.current_job_id:

                        log_step(self.current_job_id, "Dice Agent", "  [WARN] This step needs your help — couldn't confirm a real "
                                 "application form opened. Please find and click the Apply button "
                                 "yourself; the bot will resume automatically once the form appears.")
                    self._wait_for_human_to_advance_step(
                        self._page_signature(),
                        message="Bot failed — please try manually.",
                    )

            max_steps = 20

            self._just_resumed_from_human = False
            for step in range(max_steps):
                try:
                    from src.workflow.apply import skip_event, SkipRequested, check_browser_alive
                    if skip_event.is_set():
                        raise SkipRequested()
                    check_browser_alive(self.driver)
                except ImportError:
                    pass
                identity_filled = self._fill_identity_and_links(tailored_cv)
                # Also applies right after a human-assisted resume, not just on the
                # form's very first step — a slow-hydrating next page (heavy JS ATS
                # widgets, e.g. Workday) can still be rendering right when control
                # comes back to the bot. Mirrors linkedin_apply.py.
                just_resumed = self._just_resumed_from_human
                self._just_resumed_from_human = False
                if (step == 0 or just_resumed) and identity_filled == 0:
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

                            log_step(self.current_job_id, "Dice Agent", f"  [OK] LLM answered {len(answers)} additional field(s)")

                # Checked before the final-step/continue-click logic below — an
                # empty password field means the form is genuinely not done yet,
                # regardless of what the Continue button's state looks like.
                # Auto-fills a generated password instead of waiting on you to
                # type one; only falls back to waiting if the auto-fill didn't
                # actually take.
                if self._has_unfilled_password_field():
                    if self._has_auth_gate_button():
                        # A Sign-In gate, not a fresh signup — confirmed live
                        # via the same external-apply path on a GM
                        # Motorsports / Workday job that already had an
                        # account from an earlier test run of this same job:
                        # generating a brand-new random password here would
                        # be wrong (it could never match the account that
                        # was actually created), so re-use the one already
                        # generated and saved for this job instead. The
                        # email is deliberately left for you to fill — the
                        # bot has no reliable way to know whether this is
                        # really the same account/email the credentials were
                        # saved under — and the actual Sign In click is
                        # always a human action (see the login-gate
                        # exclusion in _click_continue_button).
                        saved_password = self._read_saved_password()
                        if saved_password and self._fill_all_password_fields(saved_password):
                            print("  [WARN] Reached a sign-in page — filled in your previously generated password. "
                                  "Please enter your email and click Sign In; "
                                  "the bot will resume automatically afterward.")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Dice Agent", "  [WARN] Reached a sign-in page — filled in your previously generated password. "
                                         "Please enter your email and click Sign In; "
                                         "the bot will resume automatically afterward.")
                        else:
                            print("  [WARN] Reached a sign-in page but no saved password was found for this job — "
                                  "please enter your email and password and sign in; "
                                  "the bot will resume automatically afterward.")

                            if hasattr(self, "current_job_id") and self.current_job_id:

                                log_step(self.current_job_id, "Dice Agent", "  [WARN] Reached a sign-in page but no saved password was found for this job — "
                                         "please enter your email and password and sign in; "
                                         "the bot will resume automatically afterward.")
                        self._wait_for_human_to_advance_step(
                            self._page_signature(),
                            message="Sign-in required — please enter your email and sign in.",
                        )
                        continue
                    if not self._generate_and_fill_password() or self._has_unfilled_password_field():
                        self._wait_for_password_entry()
                    continue

                if self._click_bare_agreement_gate():
                    continue

                if self._is_final_submit_step():
                    print("  [OK] Form fully filled — reached final submit step. Waiting for your review/submission.")

                    if hasattr(self, "current_job_id") and self.current_job_id:

                        log_step(self.current_job_id, "Dice Agent", "  [OK] Form fully filled — reached final submit step. Waiting for your review/submission.")
                    return True

                text_before = self._page_signature()
                clicked = self._click_continue_button()
                if not clicked:
                    # No Continue/Next button here — but that can also mean this
                    # isn't a form step at all, just another landing/launcher screen
                    # further into the flow (confirmed live via the same LinkedIn
                    # external-apply path on a FOX Careers / Workday job: after its
                    # Terms-of-Use gate, the real Workday job page shows an "Apply
                    # Manually" link — not a Continue button — which is what
                    # actually starts the real application; the same initial-apply
                    # pattern already handled once at the top of fill_form(), just
                    # needing another attempt now that we're deeper into the flow).
                    # Safe to retry here specifically because we've already ruled
                    # out both a real Continue/Next button (just above) and a real
                    # final-submit button (_is_final_submit_step(), above) — a
                    # genuine form step always has one or the other.
                    # _click_initial_apply_button() is a pure no-op (nothing
                    # to search for or click) whenever real form content is
                    # already visible — confirmed via a regression test with
                    # a synthetic form that has no Continue/Submit button at
                    # all: counting that no-op as a "retry attempt" below
                    # caused the bot to eventually escalate to
                    # _wait_for_human_to_advance_step(), which then waited
                    # forever for a page that has genuinely nothing left to
                    # click, ever. Only worth retrying — and eventually
                    # pausing for help on — when an ACTUAL search-and-click
                    # attempt is happening; if the form is already fully
                    # visible with no continue/submit button found, that's
                    # the same "nothing more to do automatically" conclusion
                    # the old code reached immediately, so reach it
                    # immediately here too.
                    already_has_form = False
                    try:
                        already_has_form = self._frame_has_form_content()
                    except Exception:
                        pass

                    # Specifically "a Continue/Save button exists and is
                    # visible but disabled" — NOT the broader "form visible,
                    # nothing clicked". Must stay this narrow: a form that is
                    # genuinely finished and has no submit control at all
                    # would hang forever if this escalated on that case too.
                    # Mirrors linkedin_apply.py.
                    has_disabled_continue = False
                    try:
                        has_disabled_continue = self._has_disabled_continue_button()
                    except Exception:
                        pass

                    has_required_empty_file = False
                    try:
                        has_required_empty_file = self._has_required_empty_file_field()
                    except Exception:
                        pass

                    if not already_has_form and self._click_initial_apply_button():
                        time.sleep(1.2)
                        # Only one automatic attempt — confirmed live on
                        # Yamaha Motor Corporation and a GM Motorsports /
                        # Workday job: letting this retry itself a couple
                        # more times either kept re-clicking the same
                        # non-advancing element loop after loop, silently, or
                        # (Yamaha) grabbed the wrong "apply"-labeled element
                        # (a site's own nav menu item) and treated whatever
                        # it revealed as a real form. Safer to hand off to
                        # you immediately after a single try than to keep
                        # guessing — the bot resumes automatically once you
                        # get it to a real form. Mirrors linkedin_apply.py.
                        self._landing_retry_count = getattr(self, "_landing_retry_count", 0) + 1
                        if self._landing_retry_count <= 0:
                            continue
                        print("  [WARN] This step needs your help — repeated automatic clicks aren't moving "
                              "the form forward. Please complete it and click Next/Continue/Submit yourself; "
                              "the bot will resume automatically once the page advances.")

                        if hasattr(self, "current_job_id") and self.current_job_id:

                            log_step(self.current_job_id, "Dice Agent", "  [WARN] This step needs your help — repeated automatic clicks aren't moving "
                                     "the form forward. Please complete it and click Next/Continue/Submit yourself; "
                                     "the bot will resume automatically once the page advances.")
                        self._wait_for_human_to_advance_step(
                            self._page_signature(),
                            message="Bot failed — please try manually.",
                        )
                        self._landing_retry_count = 0
                        continue
                    if self._has_auth_gate_button():
                        # Confirmed live on Randstad Digital / HTC Global Services jobs
                        # (Ceipal-style ATS): the portal gates the real application
                        # behind a Login/Sign-in step that only the candidate's own
                        # credentials or email verification can get past — the bot
                        # correctly refuses to click it (see _click_continue_button),
                        # but must say so plainly instead of implying the form is
                        # actually complete when only 1-2 fields got filled.
                        print("  [WARN] This application requires signing in / verifying your email on the "
                              "employer's portal — only a login gate is visible, not the real application form. "
                              "Please complete this step manually.")

                        if hasattr(self, "current_job_id") and self.current_job_id:

                            log_step(self.current_job_id, "Dice Agent", "  [WARN] This application requires signing in / verifying your email on the "
                                     "employer's portal — only a login gate is visible, not the real application form. "
                                     "Please complete this step manually.")
                    elif has_disabled_continue:
                        # A Continue/Save button exists and is visible but
                        # disabled — most likely a site (Workday confirmed
                        # live, Elevance Health) disabling it client-side
                        # because a required field is still blank, not
                        # genuinely nothing left to do. The old code treated
                        # "no clickable button + form already visible" as
                        # proof the step was done and returned immediately —
                        # silently skipping past a blocked, incomplete page
                        # (confirmed live: a custom dropdown and Yes/No
                        # question weren't recognized as fillable, so "Save
                        # and Continue" never became clickable, and the form
                        # was never actually filled). Treated exactly like
                        # the "clicked but nothing changed" case elsewhere in
                        # this loop. Mirrors linkedin_apply.py.
                        print("  [WARN] This step needs your help — a Continue/Submit button is disabled, "
                              "likely blocked by an unrecognized or unfilled required field. "
                              "Please complete it and click Next/Continue/Submit yourself; "
                              "the bot will resume automatically once the page advances.")

                        if hasattr(self, "current_job_id") and self.current_job_id:

                            log_step(self.current_job_id, "Dice Agent", "  [WARN] This step needs your help — a Continue/Submit button is disabled, "
                                     "likely blocked by an unrecognized or unfilled required field. "
                                     "Please complete it and click Next/Continue/Submit yourself; "
                                     "the bot will resume automatically once the page advances.")
                        self._wait_for_human_to_advance_step(
                            self._page_signature(),
                            message="Bot failed — please try manually.",
                        )
                        continue
                    elif has_required_empty_file:
                        # A required file field (Resume/CV) is still empty and no
                        # Continue button exists at all — confirmed live on a New
                        # York Power Authority job reached via LinkedIn's
                        # external-apply path (same underlying pattern applies
                        # here): the old code treated "no button + form visible"
                        # as proof the step was done, silently abandoning the job
                        # with a required field never filled. Mirrors the
                        # disabled-continue-button case just above.
                        print("  [WARN] This step needs your help — a required file field (e.g. Resume/CV) "
                              "is still empty and no Continue button was found. Please upload it yourself "
                              "and click Next/Continue/Submit; the bot will resume automatically once the "
                              "page advances.")

                        if hasattr(self, "current_job_id") and self.current_job_id:

                            log_step(self.current_job_id, "Dice Agent", "  [WARN] This step needs your help — a required file field (e.g. Resume/CV) "
                                     "is still empty and no Continue button was found. Please upload it yourself "
                                     "and click Next/Continue/Submit; the bot will resume automatically once the "
                                     "page advances.")
                        self._wait_for_human_to_advance_step(
                            self._page_signature(),
                            message="Bot failed — please try manually.",
                        )
                        continue
                    else:
                        print("  [OK] No further Continue/Next button found — treating current step as final.")

                        if hasattr(self, "current_job_id") and self.current_job_id:

                            log_step(self.current_job_id, "Dice Agent", "  [OK] No further Continue/Next button found — treating current step as final.")
                    return True

                self._landing_retry_count = 0

                time.sleep(1.8)
                self.wait_for_cloudflare()

                # One clean fill pass, one click, per step — never re-filling/
                # re-clicking the same step in a blind loop. If the click
                # didn't genuinely move the form forward — either nothing
                # changed at all, or it changed only to show validation
                # errors (confirmed live via the same external-apply path on
                # a Workday-based application: clicking "Save and Continue"
                # with a couple of unrecognized custom fields still blank
                # redraws the SAME page with red error banners, which IS a
                # page-signature change, so the old check here missed it and
                # kept re-attempting the same broken fields up to max_steps)
                # — stop trying to fix it automatically and wait for you to
                # finish this one step and click Next/Continue/Submit
                # yourself. The bot then resumes automatically for every step
                # after it, all the way through to the real final submit step.
                page_after_click = self._page_signature()
                validation_errors = self._detect_validation_errors()
                if page_after_click == text_before or validation_errors:
                    detail = (" Validation errors found: " + " | ".join(validation_errors[:5])) if validation_errors else \
                              " No validation-error indicators found — likely an unrecognized custom field is still blocking progress."
                    print("  [WARN] This step needs your help —" + detail +
                          " Please complete it and click Next/Continue/Submit yourself; "
                          "the bot will resume automatically once the page advances.")

                    if hasattr(self, "current_job_id") and self.current_job_id:

                        log_step(self.current_job_id, "Dice Agent", "  [WARN] This step needs your help —" + detail +
                                 " Please complete it and click Next/Continue/Submit yourself; "
                                 "the bot will resume automatically once the page advances.")
                    self._wait_for_human_to_advance_step(
                        page_after_click,
                        message="Bot failed — please try manually.",
                    )
                    continue

            print("  [WARN] Reached max auto-fill steps — leaving form for manual completion/review.")


            if hasattr(self, "current_job_id") and self.current_job_id:


                log_step(self.current_job_id, "Dice Agent", "  [WARN] Reached max auto-fill steps — leaving form for manual completion/review.")
            return True
        except Exception as e:
            print(f"Error during auto-fill: {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Dice Agent", f"Error during auto-fill: {e}")
            return False

    def wait_for_cloudflare(self):
        """Wait if a Cloudflare challenge is detected on the current page."""
        try:
            print("Checking for Cloudflare challenge...")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Dice Agent", "Checking for Cloudflare challenge...")
            for i in range(30):
                title = self.driver.title
                cf_titles = ("Just a moment", "Attention Required", "Security check", "Cloudflare")
                if any(t in title for t in cf_titles):
                    print(f"  Cloudflare detected (attempt {i+1}). Waiting for you to solve it...")

                    if hasattr(self, "current_job_id") and self.current_job_id:

                        log_step(self.current_job_id, "Dice Agent", f"  Cloudflare detected (attempt {i+1}). Waiting for you to solve it...")
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

                            log_step(self.current_job_id, "Dice Agent", f"  Cloudflare element detected (attempt {i+1}). Waiting...")
                        if i in (5, 15):
                            self.driver.refresh()
                        time.sleep(2)
                    else:
                        return True
            print("  Cloudflare did not resolve within timeout.")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Dice Agent", "  Cloudflare did not resolve within timeout.")
        except Exception as e:
            print(f"  Cloudflare check error: {e}")

            if hasattr(self, "current_job_id") and self.current_job_id:

                log_step(self.current_job_id, "Dice Agent", f"  Cloudflare check error: {e}")
        return False

    def wait_for_cloudflare_on_url(self, url: str):
        """Open a URL then wait for any Cloudflare challenge to clear."""
        self.driver.get(url)
        time.sleep(3)
        self.wait_for_cloudflare()
        time.sleep(1)

    def run(self, job_url: str, job_id: str) -> bool:
        self.current_job_id = job_id
        print(f"[Dice Agent] Starting application flow for Dice job...")

        if hasattr(self, "current_job_id") and self.current_job_id:

            log_step(self.current_job_id, "Dice Agent", f"[Dice Agent] Starting application flow for Dice job...")
        self.wait_for_cloudflare_on_url(job_url)
        return self.fill_form(
            tailored_cv=self.cv_data,
            cover_letter=self.cover_letter,
            cv_filepath=self.cv_filepath,
            job_id=job_id
        )

    def close(self):
        if self.driver:
            self.driver.quit()

