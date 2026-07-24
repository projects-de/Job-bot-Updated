"""
Standalone, isolated testbed for hardening LinkedIn EXTERNAL apply-form handling.

This file is intentionally 100% self-contained — it does not import from, or
modify, linkedin_apply.py / dice_apply.py / apply.py / indeed_apply.py. It is
not wired into the live Auto Apply pipeline. Run it directly against a single
LinkedIn job URL that is known to redirect to an external/company apply form
to observe and iterate on the improved fill logic before it is ever promoted
into production use.

Usage:
    python -m src.workflow.linkedin_test_apply "<linkedin_job_url>" [job_description.txt]

Three concrete failure modes were diagnosed from data/application_steps.log
and are specifically targeted here:
  1. "Continue" click has no real effect (form validation silently rejects
     the step) but the bot only notices after repeatedly re-filling the same
     step and gives up with no detail on *why*.
  2. Several "Continue/Next" clicks fire back-to-back with nothing filled in
     between — the page has fields, but they're built from custom ARIA
     widgets (role=combobox/listbox/radiogroup/switch) that the plain
     input/select/textarea scanner can't see at all.
  3. The very first fill pass runs before a slow-hydrating page has actually
     rendered its form, finds nothing, and concludes the application is
     already complete.
"""
import glob
import json
import os
import re
import sys
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from src.config import GROQ_API_KEY, CANDIDATE_EMAIL
from src.utils.groq_retry import retry_groq_call
from src.utils.step_logger import log_step
from src.storage.tracker import PROFILES_DIR, UPLOADS_DIR, APPLICATIONS_DIR
from src.workflow.optimize import generate_cover_letter, _lean_cv_for_prompt

# Sets an <input>/<textarea>/<select>'s value through the element's native property
# setter (bypassing React/Vue's overridden setter) so framework-controlled fields
# actually register the change — plain `.value = x` can silently fail to stick.
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

DECLINE_TO_ANSWER_KEYWORDS = ("decline", "prefer not", "do not wish", "don't wish", "not to say", "not disclose")

# Small local copies (not imports) — kept intentionally minimal since this tool
# only needs to tell "native Easy Apply" apart from "external apply" well enough
# to bail out with a clear message on the wrong kind of job.
EASY_APPLY_SELECTORS = [
    "button.jobs-apply-button[aria-label*='Easy Apply' i]",
    "xpath://button[contains(., 'Easy Apply')]",
]
EXTERNAL_APPLY_SELECTORS = [
    "xpath://button[contains(., 'Apply') and not(contains(., 'Easy Apply'))]",
    "xpath://a[contains(., 'Apply') and not(contains(., 'Easy Apply'))]",
    "a[data-control-name='jobdetails_topcard_inapply']",
    ".jobs-unified-top-card__content a.artdeco-button--primary",
]

# Validation-error indicators checked when a Continue click appears to have no
# effect — surfacing *which* field is blocking progress instead of a generic
# "stuck" warning.
_ERROR_SELECTORS = [
    "[aria-invalid='true']",
    ".error, .field-error, .has-error, .invalid-feedback, .is-invalid",
    "[class*='error' i]:not(script):not(style)",
]

CUSTOM_WIDGET_ROLE_SELECTOR = (
    "[role='combobox']:not(select), [role='listbox']:not(select), "
    "[aria-haspopup='listbox']:not(select), [role='radiogroup'], [role='switch']"
)


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


class LinkedInExternalTestApplier:
    def __init__(self, driver, cv_data: dict, cover_letter: str = "", cv_filepath: str = None, job_id: str = "linkedin-test"):
        self.driver = driver
        self.cv_data = cv_data
        self.cover_letter = cover_letter
        self.cv_filepath = cv_filepath
        self.job_id = job_id
        # Maps a synthetic id (assigned while scanning) back to the actual WebElement
        # for custom ARIA widgets, since they have no reliable name/id attribute the
        # way native inputs do — populated fresh on every _collect_unfilled_fields() call.
        self._custom_widget_map = {}

    # ------------------------------------------------------------------ logging

    def _log(self, message, level="INFO"):
        print(f"  [LI-Test] {message}")
        try:
            log_step(self.job_id, "LinkedIn Test Agent", message, level=level)
        except Exception:
            pass

    # ------------------------------------------------------------------ low-level helpers

    def _find(self, selectors, timeout=5):
        deadline = time.time() + timeout
        while time.time() < deadline:
            for sel in selectors:
                try:
                    by_type = By.XPATH if sel.startswith("xpath:") else By.CSS_SELECTOR
                    actual_sel = sel[6:] if sel.startswith("xpath:") else sel
                    for el in self.driver.find_elements(by_type, actual_sel):
                        if el.is_displayed():
                            return el
                except Exception:
                    continue
            time.sleep(0.3)
        return None

    def _click(self, selectors, description, timeout=5):
        el = self._find(selectors, timeout)
        if not el:
            return False
        try:
            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            time.sleep(0.2)
            try:
                el.click()
            except Exception:
                self.driver.execute_script("arguments[0].click();", el)
            self._log(f"Clicked {description}")
            return True
        except Exception as e:
            self._log(f"Failed to click {description}: {e}", level="WARN")
            return False

    def wait_for_cloudflare(self, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                title = (self.driver.title or "").lower()
                body = self.driver.find_element(By.TAG_NAME, "body").text.lower()
                if "checking your browser" in body or "just a moment" in title or "cloudflare" in title:
                    time.sleep(1)
                    continue
            except Exception:
                pass
            return
        self._log("Cloudflare check timed out — continuing anyway", level="WARN")

    def wait_for_page_ready(self, timeout=12):
        """
        Fix for failure mode #3: poll until at least one real interactive element
        (native or custom-widget) is visible on the page, instead of a fixed sleep
        followed immediately by a fill attempt. Slow-hydrating JS-heavy ATS forms
        (Taleo, Workday, some Greenhouse variants) can take several seconds after
        navigation before any field actually exists in the DOM — filling too early
        looks identical to "this form has nothing to fill".
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
                # Require the count to be stable across two reads 0.6s apart —
                # a page still actively hydrating will keep changing element counts.
                if count == last_count:
                    stable_reads += 1
                    if stable_reads >= 2:
                        return True
                else:
                    stable_reads = 0
                last_count = count
            time.sleep(0.6)
        self._log(f"Page-ready wait timed out after {timeout}s (last visible-element count: {last_count})", level="WARN")
        return last_count > 0

    # An external "Apply" click doesn't always land directly on the actual form —
    # confirmed live on a Dice-sourced application: Dice's own Apply link opened a
    # new tab showing the job re-posted on the company's ATS portal (Ceipal), which
    # itself still needed its own "Easy Apply"/"Apply" button clicked before the
    # real form appeared. Generic, not source-specific — the same intermediate
    # landing-page pattern is common wherever a job board hands off to a company's
    # own careers portal instead of a direct-to-form link.
    _LANDING_PAGE_APPLY_SELECTORS = [
        "xpath://button[contains(., 'Easy Apply')]", "xpath://a[contains(., 'Easy Apply')]",
        "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'apply')]",
        "xpath://a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'apply')]",
        "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'get started')]",
        "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue')]",
    ]

    # A SINGLE match here isn't enough proof of a real form — confirmed live that
    # Fastly's own nav bar has a language-selector <button role="combobox"> (a
    # legitimate accessible dropdown pattern for a totally unrelated widget) that
    # satisfies this on its own, which masked a real form sitting in a sibling
    # <iframe> from ever being found. A real application form always has several
    # fields together (name/email/phone/etc at minimum); an isolated nav widget
    # doesn't, so requiring a handful of matches together is what tells them apart.
    _FORM_FIELD_SIGNAL_SELECTOR = (
        "input[type='email'], input[type='tel'], input[type='file'], textarea, select, "
        "input[type='text'], input:not([type]), " + CUSTOM_WIDGET_ROLE_SELECTOR
    )
    _FORM_FIELD_MIN_COUNT = 3

    def _frame_has_form_content(self):
        """True if the CURRENT frame (whatever Selenium is focused on right now)
        looks like it contains a real application form — used both to check the
        top-level page and to test candidate iframes."""
        try:
            matches = [
                el for el in self.driver.find_elements(By.CSS_SELECTOR, self._FORM_FIELD_SIGNAL_SELECTOR)
                if el.is_displayed()
            ]
            return len(matches) >= self._FORM_FIELD_MIN_COUNT
        except Exception:
            return False

    def _switch_into_form_iframe_if_needed(self):
        """
        Many ATS forms (Greenhouse's embeddable job-board widget confirmed live on
        a real Fastly application, but Lever and others use the same pattern) are
        embedded inside an <iframe> on the company's own branded careers page,
        rather than the form being served directly on the visited page. Selenium
        only ever looks inside whatever frame it's currently focused on — by
        default the top-level document — so every field and button inside such an
        iframe is completely invisible to every other method in this class until
        something explicitly switches into it. That's exactly what happened on the
        Fastly job: the real Greenhouse form existed and worked fine, but every
        detection pass only ever saw the company's own marketing-site navigation
        buttons on the outer page, found nothing apply-related, and gave up.

        Only checks top-level iframes (not iframes-within-iframes) — that covers
        every real pattern seen so far and keeps this from wandering arbitrarily
        deep into unrelated embedded widgets (ads, chat widgets, trackers).
        """
        if self._frame_has_form_content():
            return True  # top-level page already has real form content — nothing to do

        try:
            candidates = self.driver.find_elements(By.TAG_NAME, "iframe")
        except Exception:
            candidates = []

        for frame_el in candidates:
            try:
                if not frame_el.is_displayed():
                    continue
                size = frame_el.size
                # Filters out tracking pixels, chat widgets, ad iframes, etc. —
                # real application-form embeds are always a substantial visible area.
                if size.get("width", 0) < 200 or size.get("height", 0) < 200:
                    continue
                # Must read this BEFORE switching — frame_el belongs to the parent
                # frame's context and becomes invalid (NoSuchElementException on any
                # attribute access) the instant switch_to.frame() moves focus into it.
                frame_id = frame_el.get_attribute("id") or ""
            except Exception:
                continue

            try:
                self.driver.switch_to.frame(frame_el)
            except Exception:
                self.driver.switch_to.default_content()
                continue

            if self._frame_has_form_content():
                self._log(f"Switched into an <iframe> containing the real application form (id='{frame_id}')")
                return True

            self.driver.switch_to.default_content()

        self._log("No form content found on the page or in any embedded iframe.", level="WARN")
        return False

    def _click_landing_page_apply_if_needed(self):
        """
        If the page already shows real form fields, does nothing. Otherwise looks
        for an Apply/Easy Apply/Continue/Get Started button and clicks it, handling
        a possible resulting new tab — mirrors the same tab-detection approach used
        everywhere else in this file (snapshot handles before the click, not after).
        """
        probe = "input:not([type='hidden']):not([type='file']), select, textarea, " + CUSTOM_WIDGET_ROLE_SELECTOR
        try:
            if any(el.is_displayed() for el in self.driver.find_elements(By.CSS_SELECTOR, probe)):
                return True
        except Exception:
            pass

        self._log("No form fields visible yet — looking for a landing-page Apply/Continue button...")
        handles_before = self.driver.window_handles
        clicked = self._click(self._LANDING_PAGE_APPLY_SELECTORS, "landing-page Apply button", timeout=10)
        if not clicked:
            self._log("No landing-page Apply button found either — proceeding as-is.", level="WARN")
            return False

        time.sleep(2.5)
        new_handles = self.driver.window_handles
        if len(new_handles) > len(handles_before):
            new_tab = [h for h in new_handles if h not in handles_before][-1]
            self.driver.switch_to.window(new_tab)
            self._log("Switched to new tab/window after landing-page Apply click.")

        self.wait_for_cloudflare()
        self.wait_for_page_ready(timeout=10)
        return True

    # ------------------------------------------------------------------ entry: open + click apply

    def open_and_click_apply(self, job_url):
        self.driver.get(job_url)
        time.sleep(3)

        if "login" in self.driver.current_url or "checkpoint" in self.driver.current_url or "authwall" in self.driver.current_url:
            self._log("LinkedIn shows a login wall — please log in manually in the opened browser (waiting up to 60s)...", level="WARN")
            deadline = time.time() + 60
            while time.time() < deadline and ("login" in self.driver.current_url or "checkpoint" in self.driver.current_url):
                time.sleep(2)
            self.driver.get(job_url)
            time.sleep(3)

        if self._find(EASY_APPLY_SELECTORS, timeout=4):
            self._log("This job uses LinkedIn's native Easy Apply, not an external form — "
                       "this tool only targets external apply-form handling. Pick a job where "
                       "LinkedIn shows 'Apply' that redirects off-site instead.", level="WARN")
            return False

        handles_before_click = self.driver.window_handles
        clicked = self._click(EXTERNAL_APPLY_SELECTORS, "external Apply button", timeout=8)
        if not clicked:
            self._log("Could not find an external Apply button on this job page.", level="WARN")
            return False

        time.sleep(2.5)
        new_handles = self.driver.window_handles
        if len(new_handles) > len(handles_before_click):
            new_tab = [h for h in new_handles if h not in handles_before_click][-1]
            self.driver.switch_to.window(new_tab)
            self._log("Switched to new tab/window for external application.")
        else:
            self._log("No new tab detected — continuing in current tab (same-tab redirect).")

        self.wait_for_cloudflare()
        return True

    # ------------------------------------------------------------------ upload helpers

    def _nearby_container_text(self, el):
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

    def _mark_uploaded(self, el):
        try:
            self.driver.execute_script("arguments[0].setAttribute('data-bot-uploaded', '1');", el)
        except Exception:
            pass

    def _try_upload_file(self, filepath, kind="resume"):
        try:
            if kind == "cover_letter":
                preferred = [
                    "input[type='file'][name*='cover' i]",
                    "input[type='file'][id*='cover' i]",
                    "input[type='file'][aria-label*='cover' i]",
                ]
            else:
                preferred = [
                    "input[type='file'][name*='resume' i]", "input[type='file'][id*='resume' i]",
                    "input[type='file'][name*='cv' i]", "input[type='file'][id*='cv' i]",
                    "input[type='file'][aria-label*='resume' i]",
                ]

            def _unfilled(elements):
                return [el for el in elements if not el.get_attribute("data-bot-uploaded")]

            for sel in preferred:
                elements = _unfilled(self.driver.find_elements(By.CSS_SELECTOR, sel))
                if elements:
                    elements[0].send_keys(os.path.abspath(filepath))
                    self._log(f"Uploaded {kind} file via: {sel}")
                    self._mark_uploaded(elements[0])
                    time.sleep(1.5)
                    return True

            own_kw = "cover" if kind == "cover_letter" else "resume"
            other_kw = "resume" if kind == "cover_letter" else "cover"
            for el in _unfilled(self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")):
                context = self._nearby_container_text(el)
                if own_kw in context and other_kw not in context:
                    el.send_keys(os.path.abspath(filepath))
                    self._log(f"Uploaded {kind} file via context-matched input ('{own_kw}')")
                    self._mark_uploaded(el)
                    time.sleep(1.5)
                    return True

            all_inputs = _unfilled(self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']"))
            if kind != "cover_letter":
                all_inputs = [el for el in all_inputs if "cover" not in self._nearby_container_text(el)]
            if all_inputs:
                all_inputs[0].send_keys(os.path.abspath(filepath))
                self._log(f"Uploaded {kind} file via first unfilled file input")
                self._mark_uploaded(all_inputs[0])
                time.sleep(1.5)
                return True
        except Exception as e:
            self._log(f"File upload failed ({kind}): {e}", level="WARN")
        return False

    # ------------------------------------------------------------------ deterministic identity fill

    def _try_fill(self, selector, value, skip_if_filled=True):
        try:
            by_type = By.XPATH if selector.startswith("/") or selector.startswith("xpath:") else By.CSS_SELECTOR
            actual_sel = selector[6:] if selector.startswith("xpath:") else selector
            for el in self.driver.find_elements(by_type, actual_sel):
                if not el.is_displayed():
                    continue
                if skip_if_filled and (el.get_attribute("value") or "").strip():
                    continue
                self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                time.sleep(0.1)
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
                el.send_keys(str(value))
                final_value = (el.get_attribute("value") or "").strip()
                if final_value != str(value).strip():
                    try:
                        self.driver.execute_script(_NATIVE_SET_VALUE_JS, el, str(value))
                    except Exception:
                        pass
                self._log(f"Filled field '{selector}' -> '{value}'")
                return True
        except Exception as e:
            self._log(f"Error filling selector '{selector}': {e}", level="WARN")
        return False

    def _try_fill_any(self, selectors, value):
        for sel in selectors:
            if self._try_fill(sel, value):
                return True
        return False

    def _select_dropdown_by_text_variants(self, select_el, variants):
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

    def _fill_identity_and_links(self, cv_data):
        full_name = cv_data.get("full_name", "")
        email = cv_data.get("email", "") or CANDIDATE_EMAIL
        phone = cv_data.get("phone", "")
        location = cv_data.get("location", "")

        name_parts = full_name.split() if full_name else []
        first_name = name_parts[0] if name_parts else ""
        last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""

        filled_count = 0

        if self._try_fill_any([
            "input[name*='full_name' i]", "input[id*='full_name' i]",
            "input[name*='fullname' i]", "input[id*='fullname' i]",
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
            "input[placeholder*='last name' i]",
        ], last_name):
            filled_count += 1

        if self._try_fill_any([
            "input[type='email']",
            "input[name*='email' i]", "input[id*='email' i]",
            "input[placeholder*='email' i]",
        ], email):
            filled_count += 1

        if email:
            verify_filled = self._try_fill_any([
                "input[name*='email' i][name*='verify' i]", "input[id*='email' i][id*='verify' i]",
                "input[name*='email' i][name*='confirm' i]", "input[id*='email' i][id*='confirm' i]",
                "input[placeholder*='verify email' i]", "input[placeholder*='confirm email' i]",
            ], email)
            if not verify_filled:
                verify_filled = self._try_fill_any(["input[type='email']"], email)
            if verify_filled:
                filled_count += 1

        if phone and self._try_fill_any([
            "input[type='tel']",
            "input[name*='phone' i]", "input[id*='phone' i]",
            "input[name*='mobile' i]", "input[id*='mobile' i]",
            "input[placeholder*='phone' i]",
        ], phone):
            filled_count += 1

        if location:
            if self._try_fill_any([
                "input[name*='location' i]:not([name*='email' i])",
                "input[id*='location' i]:not([id*='email' i])",
                "input[name*='city' i]:not([name*='email' i])",
                "input[name*='address' i]:not([name*='email' i])",
                "input[placeholder*='city' i]:not([placeholder*='email' i])",
                "input[placeholder*='address' i]:not([placeholder*='email' i]):not([type='email'])",
            ], location):
                filled_count += 1

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
                            self._log(f"State dropdown selected: {state_token}")
                            filled_count += 1
                        break

            for sel_el in self.driver.find_elements(By.CSS_SELECTOR, "select[name*='country' i], select[id*='country' i]"):
                if sel_el.is_displayed():
                    try:
                        current = Select(sel_el).first_selected_option.text.strip()
                    except Exception:
                        current = ""
                    if not current or current.lower() in ("select...", "select one", "please select", "-- select --", ""):
                        if self._select_dropdown_by_text_variants(sel_el, ["United States", "United States of America", "USA", "US"]):
                            self._log("Country dropdown selected: United States")
                            filled_count += 1
                    break

        links = cv_data.get("links", [])
        if isinstance(links, dict):
            links = [str(v) for v in links.values() if v]
        if isinstance(links, list):
            for link in links:
                l_lower = link.lower()
                if "linkedin" in l_lower and self._try_fill_any([
                    "input[name*='linkedin' i]", "input[id*='linkedin' i]", "input[placeholder*='linkedin' i]",
                ], link):
                    filled_count += 1
                elif "github" in l_lower and self._try_fill_any([
                    "input[name*='github' i]", "input[id*='github' i]", "input[placeholder*='github' i]",
                ], link):
                    filled_count += 1
                elif ("portfolio" in l_lower or "website" in l_lower) and self._try_fill_any([
                    "input[name*='portfolio' i]", "input[id*='portfolio' i]",
                    "input[name*='website' i]", "input[id*='website' i]",
                ], link):
                    filled_count += 1

        return filled_count

    def _click_consent_checkboxes(self):
        try:
            for sel in [
                "input[type='checkbox'][name*='privacy' i]",
                "input[type='checkbox'][name*='consent' i]",
                "input[type='checkbox'][name*='terms' i]",
                "input[type='checkbox'][id*='agree' i]",
            ]:
                for el in self.driver.find_elements(By.CSS_SELECTOR, sel):
                    if el.is_displayed() and not el.is_selected():
                        self.driver.execute_script("arguments[0].click();", el)
                        self._log(f"Clicked consent checkbox: {sel}")
        except Exception as e:
            self._log(f"Checkbox clicking error: {e}", level="WARN")

    def _fill_known_dropdowns(self):
        try:
            for sel in self.driver.find_elements(By.CSS_SELECTOR, "select"):
                if not sel.is_displayed():
                    continue
                aria = (sel.get_attribute("aria-label") or "").lower()
                name = (sel.get_attribute("name") or "").lower()
                text_content = (sel.text or "").lower()
                # Falls back to preceding-sibling question text when the select has no
                # name/aria-label of its own (e.g. Ladders' Gender/Race <select> — the
                # word "Gender" lives in a sibling <div>, not on the <select> at all).
                context = aria + " " + name + " " + text_content
                if not any(k in context for k in ("sponsor", "authorized", "legally", "permit", "relocat", "gender")):
                    context += " " + self._field_label(sel).lower()

                is_sponsorship = any(k in context for k in ("sponsor", "visa", "sponsorship"))
                is_auth = any(k in context for k in ("authorized", "legally", "permit"))
                is_relocate = "relocat" in context
                is_gender = "gender" in context

                sel_obj = Select(sel)
                if is_sponsorship:
                    for opt in sel_obj.options:
                        if (opt.text or "").strip().lower() in ("no", "false", "n"):
                            sel_obj.select_by_visible_text(opt.text)
                            self._log(f"Sponsorship dropdown auto-filled: {opt.text}")
                            break
                elif is_auth:
                    for opt in sel_obj.options:
                        if (opt.text or "").strip().lower() in ("yes", "true", "y"):
                            sel_obj.select_by_visible_text(opt.text)
                            self._log(f"Work Auth dropdown auto-filled: {opt.text}")
                            break
                elif is_relocate:
                    for opt in sel_obj.options:
                        if (opt.text or "").strip().lower() in ("no", "false", "n"):
                            sel_obj.select_by_visible_text(opt.text)
                            self._log(f"Relocation dropdown auto-filled: {opt.text}")
                            break
                elif is_gender:
                    for opt in sel_obj.options:
                        opt_text = (opt.text or "").lower()
                        if any(k in opt_text for k in DECLINE_TO_ANSWER_KEYWORDS):
                            sel_obj.select_by_visible_text(opt.text)
                            self._log(f"Gender/self-ID dropdown auto-filled: {opt.text}")
                            break
        except Exception as e:
            self._log(f"Dropdown auto-filling error: {e}", level="WARN")

    def _radio_label(self, el):
        gid = el.get_attribute("id") or ""
        if gid:
            try:
                label = self.driver.find_element(By.CSS_SELECTOR, f"label[for='{gid}']").text.strip()
                if label:
                    return label.lower()
            except Exception:
                pass
        # Radios with no id (common on forms like Ladders' apply4me wizard) render
        # their label as a plain sibling <label> instead of a for="id" association —
        # check the immediate wrapper for one before falling back to value/aria-label.
        try:
            wrapper = el.find_element(By.XPATH, "..")
            sib = wrapper.find_element(By.CSS_SELECTOR, "label")
            if sib.text.strip():
                return sib.text.strip().lower()
        except Exception:
            pass
        return (el.get_attribute("value") or el.get_attribute("aria-label") or "").strip().lower()

    # JS-side scan for radio groups that have NEITHER a shared `name` attribute NOR
    # per-radio ids — grouped instead by the nearest shared ancestor container that
    # holds more than one radio. This is the fix for forms (Ladders' apply4me
    # questionnaire confirmed live, but the pattern is common on other React-built
    # ATS forms too) where every Yes/No-style question radio has no name/id at all,
    # making the standard name-based grouping silently see nothing to fill. Matched
    # containers are stamped with a stable `data-bot-radio-group` attribute so later
    # passes (LLM answer application) can re-select them by CSS instead of holding
    # onto WebElement references that can go stale between calls.
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
            self._log(f"Unnamed radio-group scan error: {e}", level="WARN")
            return []

    def _stamp_synthetic_id(self, el, kind_tag):
        """
        Stamps a stable `data-bot-id` attribute on an element that has neither a
        real id nor a name, and returns a `__noid_<tag>_<n>` marker string that
        _apply_llm_field_answers() recognizes and resolves back via that attribute.
        Without this, an LLM answer for such a field has no reliable way to be
        applied back to the right element once the answer comes back asynchronously
        (Ladders' Gender/Race <select> elements are a confirmed real example).
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
            self._log(f"Error clicking unnamed radio group '{group_key}': {e}", level="WARN")
        return False

    def _fill_known_radio_groups(self):
        try:
            seen = set()
            for r in self.driver.find_elements(By.CSS_SELECTOR, "input[type='radio']"):
                name = r.get_attribute("name") or ""
                if not name or name in seen:
                    continue
                group = self.driver.find_elements(By.CSS_SELECTOR, f"input[type='radio'][name='{name}']")
                if not any(g.is_displayed() for g in group):
                    continue
                seen.add(name)
                if any(g.is_selected() for g in group):
                    continue
                context = self._nearby_container_text(group[0])
                if "relocat" in context:
                    for g in group:
                        if self._radio_label(g) == "no" and not g.is_selected():
                            self.driver.execute_script("arguments[0].click();", g)
                            self._log(f"Relocation radio auto-filled ('{name}'): No")
                            break
                elif "gender" in context or "sex" in context.split():
                    for g in group:
                        label = self._radio_label(g)
                        if any(k in label for k in DECLINE_TO_ANSWER_KEYWORDS) and not g.is_selected():
                            self.driver.execute_script("arguments[0].click();", g)
                            self._log(f"Gender/self-ID radio auto-filled ('{name}'): {label}")
                            break
        except Exception as e:
            self._log(f"Known radio-group auto-fill error: {e}", level="WARN")

        # Same set of safe, always-correct answers, but for the no-name/no-id radio
        # groups discovered via _scan_unnamed_radio_groups() — everything else
        # (background/compliance questions with no fixed right answer) is left for
        # the LLM pass in _collect_unfilled_fields()/_llm_answer_fields().
        try:
            for group in self._scan_unnamed_radio_groups():
                question = (group.get("question") or "").lower()
                options = group.get("options") or []
                key = group.get("key")
                if not key or not options:
                    continue
                if "relocat" in question:
                    if self._click_unnamed_radio_option(key, "no") or self._click_unnamed_radio_option(key, "No"):
                        self._log(f"Relocation radio auto-filled ('{key}'): No")
                elif "gender" in question or re.search(r"\bsex\b", question):
                    for opt in options:
                        if any(k in opt.lower() for k in DECLINE_TO_ANSWER_KEYWORDS):
                            if self._click_unnamed_radio_option(key, opt):
                                self._log(f"Gender/self-ID radio auto-filled ('{key}'): {opt}")
                            break
                elif ("sponsor" in question) and "visa" in question or "require sponsorship" in question:
                    if self._click_unnamed_radio_option(key, "no") or self._click_unnamed_radio_option(key, "No"):
                        self._log(f"Sponsorship radio auto-filled ('{key}'): No")
                elif "authorized to work" in question or "authorization to work" in question or \
                        ("work authorization" in question) or \
                        ("present" in question and "authorization" in question):
                    if self._click_unnamed_radio_option(key, "yes") or self._click_unnamed_radio_option(key, "Yes"):
                        self._log(f"Work authorization radio auto-filled ('{key}'): Yes")
        except Exception as e:
            self._log(f"Unnamed radio-group known-answer fill error: {e}", level="WARN")

    def _fill_date_inputs(self):
        try:
            today_iso = time.strftime("%Y-%m-%d")
            for el in self.driver.find_elements(By.CSS_SELECTOR, "input[type='date']"):
                if not el.is_displayed() or (el.get_attribute("value") or "").strip():
                    continue
                self.driver.execute_script(_NATIVE_SET_VALUE_JS, el, today_iso)
                self._log(f"Date field filled with today's date: {today_iso}")
        except Exception as e:
            self._log(f"Date field auto-fill error: {e}", level="WARN")

    def _fill_range_sliders(self, default_percent="0"):
        try:
            for el in self.driver.find_elements(By.CSS_SELECTOR, "input[type='range']"):
                if not el.is_displayed() or (el.get_attribute("value") or "").strip() == str(default_percent):
                    continue
                self.driver.execute_script(_NATIVE_SET_VALUE_JS, el, str(default_percent))
                self._log(f"Range slider set to: {default_percent}")
        except Exception as e:
            self._log(f"Range slider auto-fill error: {e}", level="WARN")

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
            aria_or_placeholder = el.get_attribute("aria-label") or el.get_attribute("placeholder") or ""
            if aria_or_placeholder:
                return aria_or_placeholder
            # No id/label/aria at all — common on no-name/no-id fields (e.g. Ladders'
            # apply4me questionnaire, same root pattern as its unnamed radios). The
            # question text is usually a sibling element rendered just before the
            # field within their shared parent, so check preceding siblings before
            # giving up with a blank label the LLM has nothing to go on.
            try:
                sib_text = self.driver.execute_script(
                    """
                    const el = arguments[0];
                    const parent = el.parentElement;
                    if (!parent) return '';
                    for (const child of parent.children) {
                        if (child === el) break;
                        const text = (child.innerText || '').trim();
                        if (text && text.length < 200) return text;
                    }
                    return '';
                    """,
                    el,
                )
                if sib_text:
                    return sib_text
            except Exception:
                pass
            return ""
        except Exception:
            return ""

    # ------------------------------------------------------------------ field detection (extended)

    def _read_custom_widget_options(self, el, role):
        """
        Best-effort peek at a custom widget's option list. Opens it (JS click),
        reads visible [role='option']/li/.option text, then closes it again via
        Escape so page state is restored. Never allowed to break the caller —
        any failure here just means this one field gets skipped, not that the
        whole scan aborts.
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

    def _collect_unfilled_fields(self):
        """
        Extended field scan: native input/select/textarea/radio/checkbox (as
        before) PLUS custom ARIA widgets (combobox/listbox/radiogroup/switch) that
        the plain native-element scan cannot see at all — this is the fix for
        failure mode #2 (several blank Continue clicks in a row on forms built
        from custom-styled components instead of native <select>/<input radio>).
        """
        fields = []
        self._custom_widget_map = {}
        seen_radio_groups = set()
        seen_custom = set()

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
                            label = self._radio_label(r)
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
                            # No id/name at all (e.g. Ladders' Gender/Race selects) —
                            # without a stable handle, an LLM answer for this field
                            # could never be located again to actually apply it. Stamp
                            # one synthetically, same approach as the unnamed radio fix.
                            select_id = self._stamp_synthetic_id(el, "select")
                        fields.append({"kind": "select", "label": self._field_label(el), "name": name, "id": select_id, "options": options})
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

            # Custom ARIA widgets — not reachable by the scan above at all.
            for el in self.driver.find_elements(By.CSS_SELECTOR, CUSTOM_WIDGET_ROLE_SELECTOR):
                try:
                    if not el.is_displayed():
                        continue
                    role = (el.get_attribute("role") or ("switch" if el.get_attribute("aria-haspopup") == "listbox" else "")).lower()
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
                    # reference — confirmed live (Dice/Ceipal) that these can go stale
                    # by the time an LLM answer comes back (the LLM round-trip takes real
                    # time, during which the site's own JS can re-render the widget).
                    # Re-querying fresh via this marker in _apply_llm_field_answers avoids
                    # that entirely.
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
            # _scan_unnamed_radio_groups) — fixes forms like Ladders' apply4me
            # questionnaire where these were previously invisible to this scan
            # entirely, so "Continue" kept clicking without ever answering them.
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

    def _detect_validation_errors(self):
        """
        Fix for failure mode #1: when a Continue click has no visible effect, scan
        for validation-error indicators instead of just giving up silently — so the
        log states exactly which field is blocking progress.
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
        # De-duplicate while preserving order
        seen = set()
        unique = []
        for f in found:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return unique

    # ------------------------------------------------------------------ LLM field answering

    def _llm_answer_fields(self, cv_data, fields, page_text):
        if not GROQ_API_KEY or not fields:
            return []
        try:
            llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0)
            fallback_llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0)
            prompt = f"""You are a job application assistant filling out an external company application form
reached via a LinkedIn "Apply" redirect (not LinkedIn's own Easy Apply flow).

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
- work authorization / eligibility / ability to present proof of identity and work
  authorization (e.g. I-9) on day one -> pick the option meaning "Yes".
- visa sponsorship required / now or in the future -> pick the option meaning "No".
- "how did you hear about us" -> pick "LinkedIn" or the closest option.
- salary / compensation expectations -> if an option means "Negotiable", choose it; otherwise leave empty ("").
- "willing to relocate" (any phrasing) -> "No".
- gender / sex / voluntary self-identification -> if an option means "Decline to self-identify" or
  "Prefer not to say", choose that one; never guess Male/Female — if no such option exists, return "".
- other yes/no eligibility, background, or compliance questions with no CV evidence either way
  (e.g. "are you a current/former government employee", "do you have/can you obtain a security
  clearance", "are you subject to any non-compete/non-solicit/NDA restrictions", criminal-history
  or background-check consent questions) -> pick the option meaning "No" — these are gating
  questions that must be answered to proceed, and "No" is the standard non-disqualifying default
  when the CV gives no reason to answer otherwise.
- "verify/confirm/re-enter email" -> repeat the exact "email" value from the CANDIDATE CV above.
- open-text questions -> 1-2 sentences using CV evidence only, never fabricate.
- for "text"/"checkbox" fields only: if a field can't be answered from the CV -> return "".
  ("select"/"radio"/"custom_select"/"custom_radio"/"custom_toggle" fields must always get one of
  their listed options per the rules above — never return "" for those kinds.)

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
            self._log(f"LLM form-answer error: {e}", level="WARN")
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
                        self._log(f"LLM-filled radio group '{group_key}' -> '{val}'")
                    else:
                        self._log(f"LLM answer '{val}' had no matching option in radio group '{group_key}'", level="WARN")
                    continue

                # Custom ARIA widget — re-query fresh via the stamped marker (see the
                # scan loop above) rather than trusting a WebElement reference held
                # since scan time, which can go stale by the time this runs.
                if fid.startswith("__custom_") and fid in self._custom_widget_map:
                    marker = self._custom_widget_map[fid]
                    matches = self.driver.find_elements(By.CSS_SELECTOR, f"[data-bot-custom-id='{marker}']")
                    if not matches:
                        self._log(f"Custom widget '{fid}' no longer present on the page — skipping.", level="WARN")
                        continue
                    el = matches[0]
                    role = (el.get_attribute("role") or "").lower()
                    if role == "radiogroup":
                        for child in el.find_elements(By.CSS_SELECTOR, "[role='radio']"):
                            label = (child.text or child.get_attribute("aria-label") or "").strip()
                            if label.lower() == str(val).strip().lower():
                                self.driver.execute_script("arguments[0].click();", child)
                                self._log(f"LLM-filled custom radio '{fid}' -> '{val}'")
                                break
                    elif role == "switch":
                        want_on = str(val).strip().lower() == "on"
                        is_on = (el.get_attribute("aria-checked") or "").lower() == "true"
                        if want_on != is_on:
                            self.driver.execute_script("arguments[0].click();", el)
                            self._log(f"LLM-filled custom toggle '{fid}' -> '{val}'")
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
                            self._log(f"LLM-filled custom dropdown '{fid}' -> '{val}'")
                        else:
                            try:
                                el.send_keys(Keys.ESCAPE)
                            except Exception:
                                pass
                            self._log(f"LLM answer '{val}' had no matching option in custom dropdown '{fid}'", level="WARN")
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
                            self._log(f"LLM-filled dropdown '{fid or fname}' -> '{val}'")
                        except Exception:
                            for opt in sel_obj.options:
                                if opt.text.strip().lower() == val.strip().lower():
                                    sel_obj.select_by_visible_text(opt.text)
                                    break
                    elif itype == "checkbox":
                        if val.lower() == "check" and not el.is_selected():
                            self.driver.execute_script("arguments[0].click();", el)
                            self._log(f"LLM-checked checkbox '{fid or fname}'")
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
                                el.send_keys(str(val))
                            except Exception:
                                pass
                            final_value = (el.get_attribute("value") or "").strip()
                            if final_value != str(val).strip():
                                try:
                                    self.driver.execute_script(_NATIVE_SET_VALUE_JS, el, str(val))
                                except Exception:
                                    pass
                            self._log(f"LLM-filled field '{fid or fname}' -> '{str(val)[:40]}'")
                    continue

                if fname:
                    radios = self.driver.find_elements(By.CSS_SELECTOR, f"input[type='radio'][name='{fname}']")
                    for r in radios:
                        if self._radio_label(r) == str(val).strip().lower() and not r.is_selected():
                            self.driver.execute_script("arguments[0].click();", r)
                            self._log(f"LLM-filled radio '{fname}' -> '{val}'")
                            break
            except Exception as e:
                self._log(f"Error applying LLM answer for '{fid or fname}': {e}", level="WARN")

    # ------------------------------------------------------------------ navigation

    def _page_signature(self):
        try:
            return self.driver.find_element(By.TAG_NAME, "body").text[:3000]
        except Exception:
            return ""

    # Final-step button text is worded very differently across ATS platforms
    # (confirmed live: Ladders uses "Apply to Job", not "Submit") — this list is
    # intentionally broad since under-matching here is the dangerous direction:
    # a final button that isn't recognized falls through to _click_continue_button(),
    # which could click it (and truly submit the application) if it also happens to
    # be a type="submit" element, instead of correctly stopping for human review.
    _FINAL_STEP_KEYWORDS = (
        "submit", "send application", "apply now", "apply to job", "apply for this job",
        "finish application", "complete application", "complete my application",
    )

    def _is_final_submit_step(self):
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

    def _click_continue_button(self):
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
                for el in self.driver.find_elements(by_type, actual_sel):
                    if not (el.is_displayed() and el.is_enabled()):
                        continue
                    text = (el.text or el.get_attribute("value") or "").strip().lower()
                    # Never click a final-step/true-submit button here, even if it
                    # incidentally matches button[type='submit'] — that must only ever
                    # be clicked by a human, after review. See _FINAL_STEP_KEYWORDS.
                    if "continue" not in text and "next" not in text and \
                            any(k in text for k in self._FINAL_STEP_KEYWORDS):
                        continue
                    self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                    time.sleep(0.2)
                    try:
                        el.click()
                    except Exception:
                        self.driver.execute_script("arguments[0].click();", el)
                    self._log(f"Clicked continue/next: '{text}'")
                    return True
            except Exception:
                pass
        return False

    def _run_all_fill_passes(self, cv_data, cover_letter):
        identity_filled = self._fill_identity_and_links(cv_data)

        if self.cv_filepath and os.path.exists(self.cv_filepath):
            self._try_upload_file(self.cv_filepath, kind="resume")

        # Only the system-generated, per-job cover letter is ever uploaded here —
        # never any other file. Falls back to typing it into a plain textarea if
        # the form has no dedicated cover-letter file field.
        uploaded_cl_file = False
        if cover_letter and self.job_id:
            cl_path = _generate_cover_letter_docx_file(cover_letter, self.job_id)
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
            answers = self._llm_answer_fields(cv_data, remaining_fields, page_text)
            if answers:
                self._apply_llm_field_answers(answers)
                self._log(f"LLM answered {len(answers)} additional field(s) (of {len(remaining_fields)} detected)")
        return identity_filled, len(remaining_fields)

    # ------------------------------------------------------------------ main loop

    def fill_form(self, cv_data, cover_letter, max_steps=8):
        # Fix for failure mode #3: don't fill until the page has actually settled.
        self.wait_for_page_ready(timeout=12)
        # Some ATS forms (Greenhouse's embeddable widget confirmed live) are inside
        # an <iframe> on the company's own page rather than served directly — must
        # happen BEFORE the landing-page-click check below, since the real
        # Apply/Continue button (if any) usually lives inside that iframe too, not
        # on the outer page.
        self._switch_into_form_iframe_if_needed()
        # Some redirects land on an intermediate job-repost/landing page (not the
        # form itself) that needs its own Apply/Continue click first — see
        # _click_landing_page_apply_if_needed for why this is needed generically.
        self._click_landing_page_apply_if_needed()

        stuck_count = 0
        for step in range(max_steps):
            identity_filled, remaining_count = self._run_all_fill_passes(cv_data, cover_letter)

            if step == 0 and identity_filled == 0 and remaining_count == 0:
                self._log("No identity fields and no other unfilled fields matched on first pass — "
                           "waiting and retrying once in case the page was still rendering...", level="WARN")
                time.sleep(2.5)
                identity_filled, remaining_count = self._run_all_fill_passes(cv_data, cover_letter)

            if self._is_final_submit_step():
                self._log("Form fully filled — reached final submit step. Waiting for your review/submission.")
                return "review_reached"

            text_before = self._page_signature()
            clicked = self._click_continue_button()
            if not clicked:
                self._log("No further Continue/Next button found — treating current step as final.")
                return "review_reached"

            time.sleep(1.8)
            self.wait_for_cloudflare()
            self.wait_for_page_ready(timeout=8)

            if self._page_signature() == text_before:
                stuck_count += 1
            else:
                stuck_count = 0

            if stuck_count == 1:
                # Fix for failure mode #1: don't just retry blindly — check for
                # validation errors and log exactly what's blocking progress, then
                # try one targeted re-fill pass before giving up for good.
                errors = self._detect_validation_errors()
                if errors:
                    self._log("Continue click had no effect — validation errors found: " + " | ".join(errors[:5]), level="WARN")
                else:
                    self._log("Continue click had no effect and no validation-error indicators found — "
                               "likely an unrecognized custom field is still blocking submission.", level="WARN")
                self._run_all_fill_passes(cv_data, cover_letter)
                clicked_retry = self._click_continue_button()
                if clicked_retry:
                    time.sleep(1.8)
                    self.wait_for_cloudflare()
                    if self._page_signature() != text_before:
                        stuck_count = 0
                        continue

            if stuck_count >= 2:
                errors = self._detect_validation_errors()
                detail = (" Validation errors still present: " + " | ".join(errors[:5])) if errors else ""
                self._log("Continue click had no visible effect after repeated attempts — "
                           "leaving form for manual completion." + detail, level="WARN")
                return "stuck"

        self._log("Reached max auto-fill steps — leaving form for manual completion/review.", level="WARN")
        return "max_steps"

    def run(self, job_url):
        if not self.open_and_click_apply(job_url):
            return "could_not_start"
        return self.fill_form(self.cv_data, self.cover_letter)


# ---------------------------------------------------------------------- standalone runner

def _latest_uploaded_resume():
    if not os.path.exists(UPLOADS_DIR):
        return None
    files = []
    for ext in ("*.pdf", "*.docx", "*.doc"):
        files.extend(glob.glob(os.path.join(UPLOADS_DIR, ext)))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def _load_cv_and_cover_letter(job_description=""):
    profile_path = os.path.join(PROFILES_DIR, "parsed_profile.json")
    if not os.path.exists(profile_path):
        raise SystemExit(
            f"No parsed CV profile found at {profile_path} — upload/parse a CV via the dashboard first."
        )
    with open(profile_path, "r", encoding="utf-8") as f:
        cv_data = json.load(f)
    print("Generating a test cover letter from the CV + job description (or a generic placeholder)...")
    cover_letter = generate_cover_letter(cv_data, job_description or "General data engineering role.")
    return cv_data, cover_letter


def _init_test_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-popup-blocking')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--no-first-run')
    options.add_argument('--no-default-browser-check')

    # Reuses the same persistent profile as the live bot so your LinkedIn login
    # session carries over — you should not need to log in again.
    user_data_dir = os.path.join(os.getcwd(), "user_data_uc")
    os.makedirs(user_data_dir, exist_ok=True)
    options.add_argument(f"--user-data-dir={user_data_dir}")

    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
        """
    })
    return driver


def run_test(job_url, job_description=""):
    cv_data, cover_letter = _load_cv_and_cover_letter(job_description)
    cv_filepath = _latest_uploaded_resume()
    if not cv_filepath:
        print("[WARN] No uploaded resume file found in data/uploads — resume upload will be skipped.")

    driver = _init_test_driver()
    applier = LinkedInExternalTestApplier(driver, cv_data, cover_letter, cv_filepath, job_id="linkedin-test")
    try:
        result = applier.run(job_url)
        print(f"\n=== TEST RESULT: {result} ===")
        print("Review the form in the browser window. Full step-by-step detail was also written to "
              "data/application_steps.log under agent 'LinkedIn Test Agent'.")
        input("Press Enter here to close the browser...")
    finally:
        driver.quit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.workflow.linkedin_test_apply \"<linkedin_job_url>\" [job_description.txt]")
        sys.exit(1)

    url = sys.argv[1]
    description = ""
    if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            description = f.read()

    run_test(url, description)
