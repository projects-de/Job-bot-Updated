import json
import os
import time
import threading
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

from src.config import USER_AGENT, CANDIDATE_EMAIL
from src.storage.tracker import log_application, save_tailored_application, JobStatus, APPLICATIONS_DIR, UPLOADS_DIR
from src.workflow.optimize import render_tailored_cv_latex
from src.workflow.indeed_apply import IndeedApplier, _is_indeed_url
from src.workflow.linkedin_apply import run_linkedin_agent
import glob

# Global events for human-in-the-loop review
review_event = threading.Event()
skip_event = threading.Event()


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
def get_chrome_major_version():
    """Retrieve the installed Google Chrome major version on Windows."""
    try:
        import winreg
        # Try Current User registry
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Google\Chrome\BLBeacon")
            version, _ = winreg.QueryValueEx(key, "version")
            winreg.CloseKey(key)
            if version:
                return int(version.split('.')[0])
        except Exception:
            pass
        # Try Local Machine registry (Wow6432Node)
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Google Chrome")
            version, _ = winreg.QueryValueEx(key, "DisplayVersion")
            winreg.CloseKey(key)
            if version:
                return int(version.split('.')[0])
        except Exception:
            pass
        # Try another Local Machine location
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Google Chrome")
            version, _ = winreg.QueryValueEx(key, "DisplayVersion")
            winreg.CloseKey(key)
            if version:
                return int(version.split('.')[0])
        except Exception:
            pass
    except Exception as e:
        print(f"Registry chrome version check failed: {e}")
    return None


class SequentialApplier:
    def __init__(self, cv_data, tailor_func, cover_letter_func):
        self.cv_data = cv_data
        self.tailor_func = tailor_func
        self.cover_letter_func = cover_letter_func
        self.driver = None

    def init_browser(self):
        if not self.driver:
            options = uc.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--disable-popup-blocking')
            
            # Onboarding & first-run suppression to bypass welcome/What's New screens
            options.add_argument('--no-first-run')
            options.add_argument('--no-default-browser-check')
            options.add_argument('--disable-features=ChromeWhatsNewUI')
            options.add_argument('--disable-whats-new-ui')
            options.add_argument('--disable-welcome-ui')
            
            # Using persistent user data directory
            user_data_dir = os.path.join(os.getcwd(), "user_data_uc")
            os.makedirs(user_data_dir, exist_ok=True)
            options.add_argument(f"--user-data-dir={user_data_dir}")
            
            major_ver = get_chrome_major_version()
            if major_ver:
                print(f"Detected installed Chrome major version: {major_ver}")
            
            try:
                if major_ver:
                    print(f"Launching undetected-chromedriver with version_main={major_ver}...")
                    self.driver = uc.Chrome(options=options, version_main=major_ver, suppress_welcome=True)
                else:
                    print("Could not detect Chrome version. Launching with default undetected-chromedriver...")
                    self.driver = uc.Chrome(options=options, suppress_welcome=True)
            except Exception as e:
                err_msg = str(e)
                print(f"Initial undetected-chromedriver launch failed: {err_msg}")
                # Fallback: parse current version from the exception error string
                import re
                match = re.search(r"Current browser version is ([\d\.]+)", err_msg)
                if match:
                    version_str = match.group(1)
                    parsed_ver = int(version_str.split('.')[0])
                    print(f"Extracted actual Chrome major version from exception: {parsed_ver}. Retrying launch...")
                    try:
                        self.driver = uc.Chrome(options=options, version_main=parsed_ver, suppress_welcome=True)
                    except Exception as retry_err:
                        print(f"Retry launch with version_main={parsed_ver} failed: {retry_err}")
                        raise retry_err
                else:
                    raise e
            
            # Deep stealth overriding
            self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                    Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                """
            })

    def _try_fill(self, selector, value):
        """Try to fill a form field matching the selector, returns True if found."""
        try:
            by_type = By.XPATH if selector.startswith("/") or selector.startswith("xpath:") else By.CSS_SELECTOR
            actual_sel = selector[6:] if selector.startswith("xpath:") else selector
            elements = self.driver.find_elements(by_type, actual_sel)
            for el in elements:
                if el.is_displayed():
                    self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                    time.sleep(0.15)
                    try:
                        el.click()
                    except:
                        pass
                    # Use keys-based clear to trigger React/Vue synthetic events reliably
                    try:
                        el.send_keys(Keys.CONTROL + "a")
                        el.send_keys(Keys.BACKSPACE)
                    except Exception:
                        el.clear()
                    time.sleep(0.1)
                    el.send_keys(str(value))
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

    def _try_upload_file(self, filepath):
        """Try to find a file input and upload the CV file."""
        try:
            file_selectors = [
                "input[type='file']",
                "input[type='file'][name*='cv' i]",
                "input[type='file'][name*='resume' i]",
                "input[type='file'][id*='cv' i]",
                "input[type='file'][id*='resume' i]",
                "input[type='file'][accept*='pdf' i]",
                "input[type='file'][accept*='doc' i]",
            ]

            for selector in file_selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    # Input type=file doesn't need to be displayed to accept keys
                    elements[0].send_keys(os.path.abspath(filepath))
                    print(f"  [OK] Uploaded CV file via: {selector}")
                    return True

            all_file_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
            if all_file_inputs:
                all_file_inputs[0].send_keys(os.path.abspath(filepath))
                print(f"  [OK] Uploaded CV file via first file input")
                return True

        except Exception as e:
            print(f"  [WARN] File upload failed: {e}")
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

    def fill_form(self, tailored_cv, cover_letter, cv_filepath=None):
        try:
            # 0. Try clicking the initial Apply button if the site requires a transition first
            self._click_initial_apply_button()

            full_name = tailored_cv.get("full_name", "")
            email = tailored_cv.get("email", "") or CANDIDATE_EMAIL
            phone = tailored_cv.get("phone", "")
            location = tailored_cv.get("location", "")
            summary = tailored_cv.get("summary", "")

            name_parts = full_name.split() if full_name else []
            first_name = name_parts[0] if name_parts else ""
            last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""

            # 1. Resume File Upload
            if cv_filepath and os.path.exists(cv_filepath):
                print(f"  Attempting file upload of: {cv_filepath}")
                self._try_upload_file(cv_filepath)
                time.sleep(1.5)

            # 2. Identity fields
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
                    "input[placeholder*='phone' i]",
                ], phone)

            if location:
                self._try_fill_any([
                    "input[name*='location' i]", "input[id*='location' i]",
                    "input[name*='city' i]", "input[id*='city' i]",
                    "input[name*='address' i]", "input[id*='address' i]",
                    "input[placeholder*='city' i]",
                ], location)

            # 3. Web links
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

            # 4. Cover Letter & Summary textareas
            if summary:
                self._try_fill_any([
                    "textarea[name*='summary' i]", "textarea[id*='summary' i]",
                    "textarea[name*='about' i]", "textarea[id*='about' i]",
                    "textarea[placeholder*='summary' i]",
                ], summary)

            if cover_letter:
                self._try_fill_any([
                    "textarea[name*='cover' i]", "textarea[id*='cover' i]",
                    "textarea[name*='letter' i]", "textarea[id*='letter' i]",
                    "textarea[name*='motivation' i]", "textarea[id*='motivation' i]",
                    "textarea[placeholder*='cover letter' i]",
                ], cover_letter)

            # 5. Autoclick general consent checkboxes
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

            # 6. Auto-resolving dropdown select questions (Sponsorship, Auth)
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

                    from selenium.webdriver.support.ui import Select as SelObject
                    sel_obj = SelObject(sel)
                    
                    if is_sponsorship:
                        # Auto select "No" to sponsorship questions
                        for opt in sel_obj.options:
                            opt_text = (opt.text or "").lower()
                            if opt_text.strip() in ("no", "false", "n"):
                                sel_obj.select_by_visible_text(opt.text)
                                print(f"  [OK] Sponsorship dropdown auto-filled: {opt.text}")
                                break
                    elif is_auth:
                        # Auto select "Yes" to work authorization questions
                        for opt in sel_obj.options:
                            opt_text = (opt.text or "").lower()
                            if opt_text.strip() in ("yes", "true", "y"):
                                sel_obj.select_by_visible_text(opt.text)
                                print(f"  [OK] Work Auth dropdown auto-filled: {opt.text}")
                                break
            except Exception as se:
                print(f"  [WARN] Dropdown auto-filling error: {se}")

            print("  [OK] Form auto-fill completed")
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

        if not job_url:
            log_application(job_id, job_title, company, JobStatus.FAILED, "Missing application URL")
            return False

        try:
            tailored_data = _load_tailored_for_job(job_id)

            if tailored_data:
                tailored_cv = tailored_data.get("cv", {})
                cover_letter = tailored_data.get("cover_letter", "")
                print(f"Using pre-generated tailored CV and cover letter for {job_title}")
            else:
                print(f"Tailoring application for {job_title} at {company}...")
                job_desc = job.get('description', '')
                tailored_cv = self.tailor_func(self.cv_data, job_desc)
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
                    self.fill_form(tailored_cv, cover_letter, cv_filepath)
                elif result == 'already_applied':
                    log_application(job_id, job_title, company, JobStatus.SUBMITTED, "Already applied via LinkedIn")
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
                    self.fill_form(tailored_cv, cover_letter, cv_filepath)
                elif result == 'failed':
                    print(f"[Indeed Agent] Could not reach review page — leaving for human review")

            else:
                # Generic path: open URL and fill form directly
                print(f"[Generic Filler] Filling form for: {job_title} at {company}")
                self.wait_for_cloudflare()
                self.fill_form(tailored_cv, cover_letter, cv_filepath)

            log_application(job_id, job_title, company, JobStatus.READY)

            print(f"--- [Job {job_id}] REVIEW REQUIRED ---")
            print(f"Role: {job_title} | Company: {company}")
            print(f"Action: Review the form in the browser, then click 'Submit Done' in the dashboard.")

            review_event.clear()
            skip_event.clear()

            # Wait for human signal or auto-detect
            while not review_event.is_set() and not skip_event.is_set():
                try:
                    current_url = self.driver.current_url.lower()
                    
                    # 1. Check for expired linkedin job
                    if 'linkedin.com' in current_url:
                        try:
                            body_el = self.driver.find_element(By.TAG_NAME, "body")
                            if body_el and "no longer accepting applications" in body_el.text.lower():
                                print(f"LinkedIn job expired for: {job_title}")
                                log_application(job_id, job_title, company, JobStatus.EXPIRED, "No longer accepting applications")
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
                    return False
                time.sleep(1.5)

            if skip_event.is_set():
                print(f"Skipped: {job_title}")
                log_application(job_id, job_title, company, JobStatus.TAILORED, "User skipped during review")
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
            return False

    def close(self):
        if self.driver:
            self.driver.quit()

def bulk_apply(jobs, cv_data, tailor_func, cover_letter_func):
    applier = SequentialApplier(cv_data, tailor_func, cover_letter_func)
    try:
        for job in jobs:
            applier.apply_to_job(job)
    finally:
        applier.close()

def signal_review_complete():
    review_event.set()

def signal_skip():
    skip_event.set()
