import json
import os
import time
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
from src.workflow.optimize import render_tailored_cv_latex
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
    def __init__(self, cv_data, cover_letter_func):
        self.cv_data = cv_data
        self.cover_letter_func = cover_letter_func
        self.driver = None

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
                    return True
        except Exception as e:
            print(f"  [DEBUG] Error filling selector '{selector}': {e}")
        return False

    def _try_fill_any(self, selectors, value):
        """Try multiple selectors in order, fill the first match."""
        for selector in selectors:
            if self._try_fill(selector, value):
                return True
        return False

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
                    "input[type='file'][accept*='pdf' i]",
                    "input[type='file'][accept*='doc' i]",
                ]

            def _unfilled(elements):
                return [el for el in elements if not el.get_attribute("data-bot-uploaded")]

            for selector in preferred_selectors:
                elements = _unfilled(self.driver.find_elements(By.CSS_SELECTOR, selector))
                if elements:
                    elements[0].send_keys(os.path.abspath(filepath))
                    self.driver.execute_script("arguments[0].setAttribute('data-bot-uploaded', '1');", elements[0])
                    time.sleep(1.5)  # let the dropzone's JS process/validate the upload before moving on
                    print(f"  [OK] Uploaded {kind} file via: {selector}")
                    return True

            # Fallback: neither semantic selector matched — many forms use generic,
            # unlabeled file inputs (e.g. a plain "Upload your resume" dropzone with
            # no name/id/aria-label at all). Fall back to the next unfilled file
            # input in DOM order: resume fields consistently render before
            # cover-letter fields, so calling this for "resume" first and
            # "cover_letter" second naturally targets the right one either way.
            all_file_inputs = _unfilled(self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']"))
            if all_file_inputs:
                all_file_inputs[0].send_keys(os.path.abspath(filepath))
                self.driver.execute_script("arguments[0].setAttribute('data-bot-uploaded', '1');", all_file_inputs[0])
                time.sleep(1.5)
                print(f"  [OK] Uploaded {kind} file via first unfilled file input")
                return True

        except Exception as e:
            print(f"  [WARN] File upload failed ({kind}): {e}")
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
                return True
        except Exception:
            pass

        print("  Form fields not visible. Searching for initial 'Apply' / 'Apply Now' button on portal...")
        
        apply_selectors = [
            "xpath://a[contains(translate(., 'APLY', 'aply'), 'apply') and not(contains(translate(., 'EASY', 'easy'), 'easy'))]",
            "xpath://button[contains(translate(., 'APLY', 'aply'), 'apply') and not(contains(translate(., 'EASY', 'easy'), 'easy'))]",
            "a[href*='/apply' i]",
            "a[href*='apply' i]",
            "button[class*='apply' i]",
            "a[class*='apply' i]",
            "a[class*='btn' i]",
            "button[class*='btn' i]"
        ]

        for sel in apply_selectors:
            try:
                by_type = By.XPATH if sel.startswith("xpath:") else By.CSS_SELECTOR
                actual_sel = sel[6:] if sel.startswith("xpath:") else sel
                elements = self.driver.find_elements(by_type, actual_sel)
                for el in elements:
                    if el.is_displayed() and el.is_enabled():
                        text = el.text.strip().replace("\n", " ")
                        href = el.get_attribute("href") or ""
                        print(f"  [OK] Found initial apply button: '{text}' (href='{href}'). Clicking it...")
                        self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                        time.sleep(0.3)
                        try:
                            el.click()
                        except Exception:
                            self.driver.execute_script("arguments[0].click();", el)
                        time.sleep(3.5) # Wait for page or form container to render
                        self.wait_for_cloudflare()
                        return True
            except Exception as e:
                print(f"  [DEBUG] Error checking initial apply selector '{sel}': {e}")
        
        print("  [WARN] No initial apply button found or clicked.")
        return False

    def _fill_identity_and_links(self, tailored_cv):
        """Deterministic pass: name, email, phone, address/location, and profile links."""
        full_name = tailored_cv.get("full_name", "")
        email = tailored_cv.get("email", "") or CANDIDATE_EMAIL
        phone = tailored_cv.get("phone", "")
        location = tailored_cv.get("location", "")
        summary = tailored_cv.get("summary", "")

        name_parts = full_name.split() if full_name else []
        first_name = name_parts[0] if name_parts else ""
        last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""

        self._try_fill_any([
            "input[name*='full_name' i]", "input[id*='full_name' i]",
            "input[name*='fullname' i]", "input[id*='fullname' i]",
            "input[name*='display_name' i]", "input[id*='display_name' i]",
            "input[name*='name' i][name*='full' i]",
            "input[placeholder*='full name' i]",
        ], full_name)

        self._try_fill_any([
            "input[name*='first_name' i]", "input[id*='first_name' i]",
            "input[name*='firstname' i]", "input[id*='firstname' i]",
            "input[name*='given' i]", "input[id*='given' i]",
            "input[placeholder*='first name' i]",
        ], first_name)

        self._try_fill_any([
            "input[name*='last_name' i]", "input[id*='last_name' i]",
            "input[name*='lastname' i]", "input[id*='lastname' i]",
            "input[name*='surname' i]", "input[id*='surname' i]",
            "input[name*='family' i]", "input[id*='family' i]",
            "input[placeholder*='last name' i]",
        ], last_name)

        self._try_fill_any([
            "input[type='email']",
            "input[name*='email' i]", "input[id*='email' i]",
            "input[placeholder*='email' i]",
        ], email)

        if phone:
            self._try_fill_any([
                "input[type='tel']",
                "input[name*='phone' i]", "input[id*='phone' i]",
                "input[name*='mobile' i]", "input[id*='mobile' i]",
                "input[name*='telephone' i]", "input[id*='telephone' i]",
                "input[placeholder*='phone' i]", "input[placeholder*='mobile' i]",
            ], phone)

        if location:
            # Covers plain "location/city" fields as well as street-address-style
            # fields (Workday/Greenhouse/Lensa commonly ask "Address"/"Address Line 1").
            # Explicitly excludes anything mentioning "email" — a placeholder like
            # "Email address" contains the substring "address" and would otherwise
            # match these selectors and get overwritten with the location value.
            self._try_fill_any([
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
            ], location)

        links = tailored_cv.get('links', [])
        if isinstance(links, list):
            for link in links:
                l_lower = link.lower()
                if 'linkedin' in l_lower:
                    self._try_fill_any([
                        "input[name*='linkedin' i]", "input[id*='linkedin' i]",
                        "input[placeholder*='linkedin' i]",
                        "input[name*='link' i][name*='in' i]",
                    ], link)
                elif 'github' in l_lower:
                    self._try_fill_any([
                        "input[name*='github' i]", "input[id*='github' i]",
                        "input[placeholder*='github' i]",
                        "input[name*='git' i]",
                    ], link)
                elif 'portfolio' in l_lower or 'website' in l_lower:
                    self._try_fill_any([
                        "input[name*='portfolio' i]", "input[id*='portfolio' i]",
                        "input[name*='website' i]", "input[id*='website' i]",
                        "input[placeholder*='website' i]",
                        "input[placeholder*='portfolio' i]",
                        "input[name*='personal' i][name*='site' i]",
                    ], link)

        if summary:
            self._try_fill_any([
                "textarea[name*='summary' i]", "textarea[id*='summary' i]",
                "textarea[name*='about' i]", "textarea[id*='about' i]",
                "textarea[placeholder*='summary' i]",
            ], summary)

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
        except Exception as ce:
            print(f"  [WARN] Checkbox clicking error: {ce}")

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

                sel_obj = Select(sel)

                if is_sponsorship:
                    for opt in sel_obj.options:
                        opt_text = (opt.text or "").lower()
                        if opt_text.strip() in ("no", "false", "n"):
                            sel_obj.select_by_visible_text(opt.text)
                            print(f"  [OK] Sponsorship dropdown auto-filled: {opt.text}")
                            break
                elif is_auth:
                    for opt in sel_obj.options:
                        opt_text = (opt.text or "").lower()
                        if opt_text.strip() in ("yes", "true", "y"):
                            sel_obj.select_by_visible_text(opt.text)
                            print(f"  [OK] Work Auth dropdown auto-filled: {opt.text}")
                            break
        except Exception as se:
            print(f"  [WARN] Dropdown auto-filling error: {se}")

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
            llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0)
            prompt = f"""You are a job application assistant filling out an external company application form
(not LinkedIn/Indeed's own flow — this is a company careers site or ATS like Greenhouse/Workday/Lensa).

CURRENT PAGE TEXT (for context):
{page_text[:1200]}

VISIBLE UNFILLED FIELDS:
{json.dumps(fields, indent=2)}

CANDIDATE CV:
{json.dumps(cv_data, indent=2)}

TASK: Return the correct value for each field based ONLY on the candidate's CV.
Rules:
- For "select"/"radio" fields, you MUST return one of the exact strings listed in that field's "options".
- For "checkbox" fields, return "check" only for standard consent/terms/privacy-policy acknowledgements; otherwise "leave unchecked".
- "years of experience" -> calculate conservatively from CV dates.
- work authorization / eligibility -> pick the option meaning "Yes".
- visa sponsorship required -> pick the option meaning "No".
- "how did you hear about us" -> pick "LinkedIn" or the closest option.
- salary / compensation expectations -> leave empty ("").
- open-text questions -> 1-2 sentences using CV evidence only, never fabricate.
- if a field can't be answered from the CV -> return "".

Return a JSON array ONLY, no explanation, matching each field by "id" (or "name" if it has no id):
[{{"id": "field_id_or_empty", "name": "field_name_or_empty", "value": "answer"}}, ...]"""

            response = retry_groq_call(lambda: llm.invoke([HumanMessage(content=prompt)]))
            text = response.content.strip()
            if "[" in text and "]" in text:
                text = text[text.find("["):text.rfind("]") + 1]
            return json.loads(text)
        except Exception as e:
            print(f"  [WARN] LLM form-answer error: {e}")
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
                        except Exception:
                            for opt in sel_obj.options:
                                if opt.text.strip().lower() == val.strip().lower():
                                    sel_obj.select_by_visible_text(opt.text)
                                    break
                    elif itype == "checkbox":
                        if val.lower() == "check" and not el.is_selected():
                            self.driver.execute_script("arguments[0].click();", el)
                            print(f"  [OK] LLM-checked checkbox '{fid or fname}'")
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
                    return True
            except Exception:
                pass
        return False

    def fill_form(self, tailored_cv, cover_letter, cv_filepath=None, job_id=None):
        """
        Fill an external application form. Handles multi-step wizards (e.g. Lensa,
        Workday, Greenhouse) by looping: fill whatever is visible on the current
        step, then click Continue/Next and repeat, until a final Submit step is
        reached (left for the human to review and submit) or the form gets stuck.
        """
        try:
            self._click_initial_apply_button()

            max_steps = 6
            stuck_count = 0

            for step in range(max_steps):
                self._fill_identity_and_links(tailored_cv)

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

                page_text = self._page_signature()
                remaining_fields = self._collect_unfilled_fields()
                if remaining_fields:
                    answers = self._llm_answer_fields(tailored_cv, remaining_fields, page_text)
                    if answers:
                        self._apply_llm_field_answers(answers)
                        print(f"  [OK] LLM answered {len(answers)} additional field(s)")

                if self._is_final_submit_step():
                    print("  [OK] Form fully filled — reached final submit step. Waiting for your review/submission.")
                    return True

                text_before = self._page_signature()
                clicked = self._click_continue_button()
                if not clicked:
                    print("  [OK] No further Continue/Next button found — treating current step as final.")
                    return True

                time.sleep(1.8)
                self.wait_for_cloudflare()

                if self._page_signature() == text_before:
                    stuck_count += 1
                else:
                    stuck_count = 0

                if stuck_count >= 2:
                    print("  [WARN] Continue click had no visible effect after repeated attempts — leaving form for manual completion.")
                    return True

            print("  [WARN] Reached max auto-fill steps — leaving form for manual completion/review.")
            return True
        except Exception as e:
            print(f"Error during auto-fill: {e}")
            return False

    def wait_for_cloudflare(self):
        """Wait if a Cloudflare challenge is detected on the current page."""
        try:
            print("Checking for Cloudflare challenge...")
            for i in range(30):
                title = self.driver.title
                cf_titles = ("Just a moment", "Attention Required", "Security check", "Cloudflare")
                if any(t in title for t in cf_titles):
                    print(f"  Cloudflare detected (attempt {i+1}). Waiting for you to solve it...")
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
                        if i in (5, 15):
                            self.driver.refresh()
                        time.sleep(2)
                    else:
                        return True
            print("  Cloudflare did not resolve within timeout.")
        except Exception as e:
            print(f"  Cloudflare check error: {e}")
        return False

    def wait_for_cloudflare_on_url(self, url: str):
        """Open a URL then wait for any Cloudflare challenge to clear."""
        self.driver.get(url)
        time.sleep(3)
        self.wait_for_cloudflare()
        time.sleep(1)

    def apply_to_job(self, job):
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
            else:
                print(f"Generating cover letter for {job_title} at {company}...")
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
            
            # Prioritize the originally uploaded high-quality PDF/DOCX resume
            original_resume_path = _get_latest_uploaded_resume()
            if original_resume_path:
                cv_filepath = original_resume_path
                print(f"Using high-quality original resume for upload: {os.path.basename(cv_filepath)}")
            else:
                cv_filepath = _generate_cv_text_file(tailored_cv, job_id)
                print(f"Fallback to text tailored resume: {os.path.basename(cv_filepath)}")

            self.init_browser()
            self.wait_for_cloudflare_on_url(job_url)

            # Record window handles to manage external redirections/new tabs
            original_handles = self.driver.window_handles
            main_handle = self.driver.current_window_handle

            # ── Route by job source ───────────────────────────────────────────
            if job_source == 'linkedin' or 'linkedin.com' in job_url.lower():
                print(f"[LinkedIn Agent] Starting LangGraph agent for: {job_title} at {company}")
                result = run_linkedin_agent(
                    driver=self.driver,
                    job_url=job_url,
                    cv_data=tailored_cv,
                    cover_letter=cover_letter,
                    cv_filepath=cv_filepath,
                )
                if result == 'external':
                    # LinkedIn job redirects to external site — wait and switch to newly opened tab if any
                    print(f"[Generic Filler] LinkedIn external apply for: {job_title}")
                    time.sleep(3.5)
                    new_handles = self.driver.window_handles
                    if len(new_handles) > len(original_handles):
                        new_tab = [h for h in new_handles if h not in original_handles][-1]
                        self.driver.switch_to.window(new_tab)
                        print("  Switched to new tab/window for external application.")
                    else:
                        print("  No new tab detected, continuing in current tab.")
                    
                    self.wait_for_cloudflare()
                    self.fill_form(tailored_cv, cover_letter, cv_filepath, job_id=job_id)
                elif result == 'already_applied':
                    log_application(job_id, job_title, company, JobStatus.SUBMITTED, "Already applied via LinkedIn")
                    review_deadline_epoch["value"] = None
                    return True
                elif result == 'failed':
                    print(f"[LinkedIn Agent] Could not complete — leaving browser open for manual apply")

            elif job_source == 'indeed' or _is_indeed_url(job_url):
                print(f"[Indeed Agent] Starting for: {job_title} at {company}")
                agent = IndeedApplier(
                    driver=self.driver,
                    cv_data=tailored_cv,
                    cover_letter=cover_letter,
                    cv_filepath=cv_filepath,
                )
                result = agent.run(job_url)
                if result == 'external_redirect':
                    print(f"[Generic Filler] Indeed external apply for: {job_title}")
                    time.sleep(3.5)
                    new_handles = self.driver.window_handles
                    if len(new_handles) > len(original_handles):
                        new_tab = [h for h in new_handles if h not in original_handles][-1]
                        self.driver.switch_to.window(new_tab)
                        print("  Switched to new tab/window for external Indeed redirect.")
                    else:
                        print("  No new tab detected, continuing in current tab.")
                    
                    self.wait_for_cloudflare()
                    self.fill_form(tailored_cv, cover_letter, cv_filepath, job_id=job_id)
                elif result == 'failed':
                    print(f"[Indeed Agent] Could not reach review page — leaving for human review")

            else:
                # Generic path: open URL and fill form directly
                print(f"[Generic Filler] Filling form for: {job_title} at {company}")
                self.wait_for_cloudflare()
                self.fill_form(tailored_cv, cover_letter, cv_filepath, job_id=job_id)

            log_application(job_id, job_title, company, JobStatus.READY)

            print(f"--- [Job {job_id}] REVIEW REQUIRED ---")
            print(f"Role: {job_title} | Company: {company}")
            print(f"Action: Review the form in the browser, then click 'Submit Done' in the dashboard.")

            review_event.clear()
            skip_event.clear()

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
                                log_application(job_id, job_title, company, JobStatus.EXPIRED, "No longer accepting applications")
                                review_deadline_epoch["value"] = None
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

                    # URL change detection
                    if any(word in current_url for word in ['success', 'thank', 'completed', 'submitted', 'post-apply']):
                        is_submitted = True
                    else:
                        try:
                            # JS event listener check
                            js_detector = """
                            if (!window.__bot_detector_injected) {
                                window.__bot_detector_injected = true;
                                window.__bot_form_submitted = false;
                                document.addEventListener('click', function(e) {
                                    let t = (e.target.innerText || e.target.value || e.target.getAttribute('aria-label') || '').toLowerCase();
                                    if (t.includes('submit')) {
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
                                is_submitted = True
                        except Exception:
                            pass

                    if is_submitted:
                        print(f"Auto-detected submission for: {job_title}")
                        review_event.set()
                        break

                    # Standard check to ensure browser is still open
                    self.driver.title
                except Exception as e:
                    print(f"Browser closed for: {job_title}")
                    log_application(job_id, job_title, company, JobStatus.TAILORED, "User closed browser before submit")
                    review_deadline_epoch["value"] = None
                    return False
                time.sleep(1.5)

            review_deadline_epoch["value"] = None

            if timed_out:
                print(f"Review timed out after {REVIEW_TIMEOUT_SECONDS}s without submission: {job_title}")
                log_application(job_id, job_title, company, JobStatus.TAILORED, f"Timed out after {REVIEW_TIMEOUT_SECONDS}s without submission")
                try:
                    self.driver.quit()
                except Exception:
                    pass
                self.driver = None
                return False

            if skip_event.is_set():
                print(f"Skipped: {job_title}")
                log_application(job_id, job_title, company, JobStatus.SKIPPED, "User skipped during review")
                # Clean up secondary window handles and return to primary
                try:
                    current_handle = self.driver.current_window_handle
                    if len(self.driver.window_handles) > 1 and current_handle != original_handles[0]:
                        self.driver.close()
                        self.driver.switch_to.window(original_handles[0])
                except Exception:
                    pass
                return False

            log_application(job_id, job_title, company, JobStatus.SUBMITTED)
            print(f"[OK] Submitted: {job_title}")

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

        except Exception as e:
            print(f"Failed to apply to {job_title}: {e}")
            log_application(job_id, job_title, company, JobStatus.FAILED, str(e))
            review_deadline_epoch["value"] = None
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
