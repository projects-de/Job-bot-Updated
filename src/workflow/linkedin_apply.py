"""
LinkedIn Easy Apply agent built with LangGraph.

Graph:
  START -> open_job -> check_login -> [wait_login ->] click_easy_apply
        -> fill_step ⟷ advance (loop) -> review -> END
                                      -> handle_error -> END

The LLM (Groq) is used only where intelligence is genuinely needed:
  - Screening questions: reads the question, answers from CV evidence
  - Step-type detection fallback when selectors are ambiguous
Everything else (contact info, resume, cover letter) uses deterministic
Selenium selectors for speed and reliability.
"""

import json
import os
import re
import secrets
import string
import time
from src.utils.step_logger import log_step
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.webdriver.common.keys import Keys

from src.config import CANDIDATE_EMAIL, GROQ_API_KEY
from src.utils.groq_retry import retry_groq_call
from src.storage.tracker import APPLICATIONS_DIR
from src.workflow.optimize import _lean_cv_for_prompt


# ── State ─────────────────────────────────────────────────────────────────────

class LinkedInState(TypedDict):
    driver: Any          # Selenium WebDriver — not serialized, in-memory only
    job_url: str
    cv_data: dict
    cover_letter: str
    cv_filepath: str
    job_id: str
    step_count: int
    max_steps: int
    login_done: bool
    result: str          # 'review_reached' | 'external' | 'failed' | 'already_applied'
    error: str
    stuck_count: int      # consecutive Next-clicks that produced no visible modal change


# ── Low-level Selenium helpers ────────────────────────────────────────────────

# LinkedIn's current Easy Apply modal is a web component rendered inside an
# OPEN shadow root (host: #interop-outlet). Regular find_element(s) calls never
# cross that boundary, so every CSS query below is routed through this JS
# helper, which searches both the main document and that shadow root.
_SHADOW_QUERY_JS = """
    const selector = arguments[0];
    const textFilter = arguments[1];
    const roots = [document];
    const host = document.querySelector('#interop-outlet');
    if (host && host.shadowRoot) roots.push(host.shadowRoot);
    let results = [];
    for (const root of roots) {
        try { results = results.concat(Array.from(root.querySelectorAll(selector))); } catch (e) {}
    }
    if (textFilter) {
        const tf = textFilter.toLowerCase();
        results = results.filter(el => (el.textContent || '').toLowerCase().includes(tf));
    }
    return results;
"""


def _dom_query(driver, css_selector: str, text_contains: str = None) -> list:
    """
    CSS-selector query that also reaches into LinkedIn's Easy Apply shadow-DOM
    web component, which driver.find_elements(By.CSS_SELECTOR, ...) can't see.
    Optionally filters results to elements whose text contains `text_contains`.
    """
    try:
        return driver.execute_script(_SHADOW_QUERY_JS, css_selector, text_contains) or []
    except Exception:
        return []


def _find(driver, selectors: list, timeout: float = 6):
    """Return first visible element matching any selector (CSS, shadow-aware CSS, or XPath), or None."""
    deadline = time.time() + timeout
    print(f"  [DEBUG] Searching for selectors: {selectors}")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", f"  [DEBUG] Searching for selectors: {selectors}")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", f"  [DEBUG] Searching for selectors: {selectors}")

    except Exception:

        pass
    while time.time() < deadline:
        for sel in selectors:
            try:
                if sel.startswith("/") or sel.startswith("xpath:"):
                    actual_sel = sel[6:] if sel.startswith("xpath:") else sel
                    elements = driver.find_elements(By.XPATH, actual_sel)
                else:
                    elements = _dom_query(driver, sel)
                if elements:
                    print(f"  [DEBUG] Selector '{sel}' matched {len(elements)} elements.")

                    try:

                        if "state" in locals() and state.get("job_id"):

                            log_step(state["job_id"], "LinkedIn Agent", f"  [DEBUG] Selector '{sel}' matched {len(elements)} elements.")

                        elif "job_id" in locals() and job_id:

                            log_step(job_id, "LinkedIn Agent", f"  [DEBUG] Selector '{sel}' matched {len(elements)} elements.")

                    except Exception:

                        pass
                    for i, el in enumerate(elements):
                        try:
                            disp = el.is_displayed()
                            enabled = el.is_enabled()
                            text = el.text[:50].replace("\n", " ") if disp else ""
                            print(f"    - Element {i}: displayed={disp}, enabled={enabled}, text='{text}'")

                            try:

                                if "state" in locals() and state.get("job_id"):

                                    log_step(state["job_id"], "LinkedIn Agent", f"    - Element {i}: displayed={disp}, enabled={enabled}, text='{text}'")

                                elif "job_id" in locals() and job_id:

                                    log_step(job_id, "LinkedIn Agent", f"    - Element {i}: displayed={disp}, enabled={enabled}, text='{text}'")

                            except Exception:

                                pass
                            if disp:
                                return el
                        except Exception as ee:
                            print(f"    - Element {i} error: {ee}")

                            try:

                                if "state" in locals() and state.get("job_id"):

                                    log_step(state["job_id"], "LinkedIn Agent", f"    - Element {i} error: {ee}")

                                elif "job_id" in locals() and job_id:

                                    log_step(job_id, "LinkedIn Agent", f"    - Element {i} error: {ee}")

                            except Exception:

                                pass
            except Exception as e:
                print(f"  [DEBUG] Error checking selector '{sel}': {e}")

                try:

                    if "state" in locals() and state.get("job_id"):

                        log_step(state["job_id"], "LinkedIn Agent", f"  [DEBUG] Error checking selector '{sel}': {e}")

                    elif "job_id" in locals() and job_id:

                        log_step(job_id, "LinkedIn Agent", f"  [DEBUG] Error checking selector '{sel}': {e}")

                except Exception:

                    pass
        time.sleep(0.35)
    print("  [DEBUG] No visible element found for any selector.")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "  [DEBUG] No visible element found for any selector.")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "  [DEBUG] No visible element found for any selector.")

    except Exception:

        pass
    return None


def _safe_fill(driver, el, value: str) -> bool:
    try:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
        time.sleep(0.1)
        try:
            el.click()
        except Exception:
            pass
        # Clear field using backspace simulation to trigger React/SPA state updates reliably
        try:
            el.send_keys(Keys.CONTROL + "a")
            el.send_keys(Keys.BACKSPACE)
        except Exception:
            driver.execute_script("arguments[0].value = '';", el)
        time.sleep(0.1)
        el.send_keys(str(value))
        return True
    except Exception:
        return False


def _fill_location_typeahead(driver, el, location_val: str) -> bool:
    """Type city name, wait for LinkedIn autocomplete typeahead, and select first option."""
    try:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
        time.sleep(0.2)
        el.click()
        time.sleep(0.1)
        
        # Clear field using backspace simulation to trigger React state updates
        el.send_keys(Keys.CONTROL + "a")
        el.send_keys(Keys.BACKSPACE)
        time.sleep(0.2)
        
        # Type the first part of the location (e.g. "Islamabad" instead of full "Islamabad, Pakistan")
        # because typeaheads prefer cities without full suffixes to trigger search results
        search_part = location_val.split(",")[0].strip()
        el.send_keys(search_part)
        print(f"    Typed location search: '{search_part}'")

        try:

            if "state" in locals() and state.get("job_id"):

                log_step(state["job_id"], "LinkedIn Agent", f"    Typed location search: '{search_part}'")

            elif "job_id" in locals() and job_id:

                log_step(job_id, "LinkedIn Agent", f"    Typed location search: '{search_part}'")

        except Exception:

            pass
        time.sleep(1.5)  # Wait for results to render
        
        # Look for autocomplete options
        option_selectors = [
            "div[role='option']",
            "li[role='option']",
            ".artdeco-typeahead__result",
            ".artdeco-typeahead__results li",
            "ul.artdeco-typeahead__results li"
        ]
        
        for sel in option_selectors:
            options = driver.find_elements(By.CSS_SELECTOR, sel)
            for opt in options:
                if opt.is_displayed():
                    opt_text = opt.text.replace("\n", " ")
                    print(f"    [OK] Selecting location option: '{opt_text}'")

                    try:

                        if "state" in locals() and state.get("job_id"):

                            log_step(state["job_id"], "LinkedIn Agent", f"    [OK] Selecting location option: '{opt_text}'")

                        elif "job_id" in locals() and job_id:

                            log_step(job_id, "LinkedIn Agent", f"    [OK] Selecting location option: '{opt_text}'")

                    except Exception:

                        pass
                    driver.execute_script("arguments[0].click();", opt)
                    time.sleep(1)
                    return True
                    
        # If option selectors didn't match, try pressing Enter or Down + Enter as fallback
        print("    [WARN] Autocomplete dropdown selector not matched, attempting Keys.DOWN + Keys.ENTER fallback...")

        try:

            if "state" in locals() and state.get("job_id"):

                log_step(state["job_id"], "LinkedIn Agent", "    [WARN] Autocomplete dropdown selector not matched, attempting Keys.DOWN + Keys.ENTER fallback...")

            elif "job_id" in locals() and job_id:

                log_step(job_id, "LinkedIn Agent", "    [WARN] Autocomplete dropdown selector not matched, attempting Keys.DOWN + Keys.ENTER fallback...")

        except Exception:

            pass
        el.send_keys(Keys.DOWN)
        time.sleep(0.3)
        el.send_keys(Keys.ENTER)
        time.sleep(1)
        return True
    except Exception as e:
        print(f"    [WARN] Location typeahead error: {e}")

        try:

            if "state" in locals() and state.get("job_id"):

                log_step(state["job_id"], "LinkedIn Agent", f"    [WARN] Location typeahead error: {e}")

            elif "job_id" in locals() and job_id:

                log_step(job_id, "LinkedIn Agent", f"    [WARN] Location typeahead error: {e}")

        except Exception:

            pass
        return False


def _fill_any(driver, selectors: list, value: str, skip_if_filled=True) -> bool:
    """Fill first visible matching element. Skips if already has a value."""
    for sel in selectors:
        try:
            for el in _dom_query(driver, sel):
                if not el.is_displayed():
                    continue
                if skip_if_filled and (el.get_attribute("value") or "").strip():
                    continue
                # If it's a city or location field, handle autocomplete typeahead
                is_loc = any(k in sel.lower() or k in (el.get_attribute("id") or "").lower() or k in (el.get_attribute("name") or "").lower() for k in ("city", "location"))
                if is_loc:
                    if _fill_location_typeahead(driver, el, value):
                        return True
                elif _safe_fill(driver, el, value):
                    return True
        except Exception:
            pass
    return False


def _click(driver, selectors: list, label: str = "", timeout: float = 8) -> bool:
    el = _find(driver, selectors, timeout)
    if not el:
        print(f"  [DEBUG] Click aborted: Selector not found for: {label}")

        try:

            if "state" in locals() and state.get("job_id"):

                log_step(state["job_id"], "LinkedIn Agent", f"  [DEBUG] Click aborted: Selector not found for: {label}")

            elif "job_id" in locals() and job_id:

                log_step(job_id, "LinkedIn Agent", f"  [DEBUG] Click aborted: Selector not found for: {label}")

        except Exception:

            pass
        return False
    try:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
        time.sleep(0.3)
        el.click()
        if label:
            print(f"    [OK] Clicked: {label}")

            try:

                if "state" in locals() and state.get("job_id"):

                    log_step(state["job_id"], "LinkedIn Agent", f"    [OK] Clicked: {label}")

                elif "job_id" in locals() and job_id:

                    log_step(job_id, "LinkedIn Agent", f"    [OK] Clicked: {label}")

            except Exception:

                pass
        time.sleep(1.5)
        return True
    except ElementClickInterceptedException as ec:
        print(f"  [DEBUG] Click intercepted for '{label}': {ec}. Retrying via execute_script...")

        try:

            if "state" in locals() and state.get("job_id"):

                log_step(state["job_id"], "LinkedIn Agent", f"  [DEBUG] Click intercepted for '{label}': {ec}. Retrying via execute_script...")

            elif "job_id" in locals() and job_id:

                log_step(job_id, "LinkedIn Agent", f"  [DEBUG] Click intercepted for '{label}': {ec}. Retrying via execute_script...")

        except Exception:

            pass
        try:
            driver.execute_script("arguments[0].click();", el)
            if label:
                print(f"    [OK] Clicked (stealth retry) for: {label}")

                try:

                    if "state" in locals() and state.get("job_id"):

                        log_step(state["job_id"], "LinkedIn Agent", f"    [OK] Clicked (stealth retry) for: {label}")

                    elif "job_id" in locals() and job_id:

                        log_step(job_id, "LinkedIn Agent", f"    [OK] Clicked (stealth retry) for: {label}")

                except Exception:

                    pass
            time.sleep(1.5)
            return True
        except Exception as e2:
            print(f"  [DEBUG] Stealth retry click failed for '{label}': {e2}")

            try:

                if "state" in locals() and state.get("job_id"):

                    log_step(state["job_id"], "LinkedIn Agent", f"  [DEBUG] Stealth retry click failed for '{label}': {e2}")

                elif "job_id" in locals() and job_id:

                    log_step(job_id, "LinkedIn Agent", f"  [DEBUG] Stealth retry click failed for '{label}': {e2}")

            except Exception:

                pass
            return False
    except Exception as e:
        print(f"  [DEBUG] Click failed for '{label}' with exception: {e}")

        try:

            if "state" in locals() and state.get("job_id"):

                log_step(state["job_id"], "LinkedIn Agent", f"  [DEBUG] Click failed for '{label}' with exception: {e}")

            elif "job_id" in locals() and job_id:

                log_step(job_id, "LinkedIn Agent", f"  [DEBUG] Click failed for '{label}' with exception: {e}")

        except Exception:

            pass
        return False


def _click_shadow_by_text(driver, tag: str, texts: list, label: str = "") -> bool:
    """Find and click a shadow-DOM-aware element by tag + text content match (case-insensitive)."""
    for text in texts:
        for el in _dom_query(driver, tag, text_contains=text):
            try:
                if not (el.is_displayed() and el.is_enabled()):
                    continue
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                time.sleep(0.2)
                try:
                    el.click()
                except Exception:
                    driver.execute_script("arguments[0].click();", el)
                if label:
                    print(f"    [OK] Clicked (shadow-aware text match): {label} ('{text}')")

                    try:

                        if "state" in locals() and state.get("job_id"):

                            log_step(state["job_id"], "LinkedIn Agent", f"    [OK] Clicked (shadow-aware text match): {label} ('{text}')")

                        elif "job_id" in locals() and job_id:

                            log_step(job_id, "LinkedIn Agent", f"    [OK] Clicked (shadow-aware text match): {label} ('{text}')")

                    except Exception:

                        pass
                time.sleep(1.5)
                return True
            except Exception:
                pass
    return False


MODAL_SELECTORS = [".jobs-easy-apply-modal", "div[role='dialog']", ".artdeco-modal", "[aria-modal='true']", "dialog"]


def _modal_text(driver) -> str:
    """Extract visible text from the Easy Apply modal (capped at 2000 chars)."""
    for sel in MODAL_SELECTORS:
        try:
            for el in _dom_query(driver, sel):
                if el.is_displayed():
                    return el.text[:2000]
        except Exception:
            pass
    return ""


def _modal_fields(driver) -> list:
    """
    Extract all visible, unfilled form fields from the modal.
    Returns list of dicts: {label, type, id, name, current_value}
    """
    fields = []
    modal = None
    for sel in MODAL_SELECTORS:
        try:
            for el in _dom_query(driver, sel):
                if el.is_displayed():
                    modal = el
                    break
        except Exception:
            pass
        if modal:
            break
    if not modal:
        return fields

    try:
        inputs = modal.find_elements(
            By.CSS_SELECTOR,
            "input:not([type='hidden']):not([type='file']), select, textarea"
        )
        for inp in inputs:
            try:
                if not inp.is_displayed():
                    continue
                itype = inp.get_attribute("type") or inp.tag_name
                iid = inp.get_attribute("id") or ""
                iname = inp.get_attribute("name") or ""
                current = inp.get_attribute("value") or ""

                label_text = ""
                if iid:
                    try:
                        lbl = modal.find_element(By.CSS_SELECTOR, f"label[for='{iid}']")
                        label_text = lbl.text.strip()
                    except Exception:
                        pass
                if not label_text:
                    try:
                        label_text = inp.find_element(By.XPATH, "ancestor::label").text.strip()
                    except Exception:
                        pass
                if not label_text:
                    label_text = inp.get_attribute("aria-label") or inp.get_attribute("placeholder") or ""

                fields.append({
                    "label": label_text,
                    "type": itype,
                    "id": iid,
                    "name": iname,
                    "current_value": current,
                })
            except Exception:
                pass
    except Exception:
        pass
    return fields


# ── Login detection ───────────────────────────────────────────────────────────

def _is_login_wall(driver) -> bool:
    try:
        # Check if we are already logged in via prominent navbar or profile indicators
        logged_in_selectors = [
            ".global-nav",
            ".global-nav__me-photo",
            ".global-nav__me-menu-trigger",
            "#global-nav-typeahead",
            "a[href*='/in/']"
        ]
        for sel in logged_in_selectors:
            try:
                for el in driver.find_elements(By.CSS_SELECTOR, sel):
                    if el.is_displayed():
                        print(f"  [DEBUG] Logged-in indicator found: {sel}")

                        try:

                            if "state" in locals() and state.get("job_id"):

                                log_step(state["job_id"], "LinkedIn Agent", f"  [DEBUG] Logged-in indicator found: {sel}")

                            elif "job_id" in locals() and job_id:

                                log_step(job_id, "LinkedIn Agent", f"  [DEBUG] Logged-in indicator found: {sel}")

                        except Exception:

                            pass
                        return False
            except Exception:
                pass

        url = driver.current_url.lower()
        print(f"  [DEBUG] Login check URL: {url}")

        try:

            if "state" in locals() and state.get("job_id"):

                log_step(state["job_id"], "LinkedIn Agent", f"  [DEBUG] Login check URL: {url}")

            elif "job_id" in locals() and job_id:

                log_step(job_id, "LinkedIn Agent", f"  [DEBUG] Login check URL: {url}")

        except Exception:

            pass
        if any(k in url for k in ("login", "signin", "sign-in", "authwall", "checkpoint", "uas/oauth")):
            return True
        # Treat Google OAuth page as still-in-login (user may have clicked "Continue with Google")
        if "accounts.google.com" in url or "google.com/o/oauth" in url:
            return True
        login_selectors = [
            "#username", "#password",
            "form[action*='login']",
            "button[aria-label*='Sign in with Google' i]",
            ".sign-in-modal",
            "a[data-tracking-control-name*='sign-in' i]",
        ]
        for sel in login_selectors:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            if any(e.is_displayed() for e in els):
                return True
        body = driver.find_element(By.TAG_NAME, "body").text.lower()
        return ("sign in" in body or "log in" in body) and "new to linkedin" in body
    except Exception:
        return False


# ── Review page detection ─────────────────────────────────────────────────────

def _is_review_page(driver) -> bool:
    review_selectors = [
        "button[aria-label*='Submit application' i]",
        "button[aria-label='Review your application']",
        "button[aria-label*='submit your application' i]",
    ]
    for sel in review_selectors:
        try:
            if any(e.is_displayed() for e in _dom_query(driver, sel)):
                return True
        except Exception:
            pass
    try:
        text = _modal_text(driver).lower()
        return "submit application" in text or "review your application" in text
    except Exception:
        return False


# ── LLM: intelligent screening question answering ─────────────────────────────

def _llm_answer_fields(cv_data: dict, fields: list, modal_context: str) -> list:
    """
    Ask Groq to read visible form fields and return answers based on CV data.
    Returns: [{id, value}, ...]
    Only called when there are unfilled fields that need intelligent answers.
    """
    if not GROQ_API_KEY or not fields:
        return []
    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="openai/gpt-oss-120b",
            temperature=0,
            reasoning_effort="low",
            max_tokens=3000,
        )
        fallback_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="openai/gpt-oss-20b",
            temperature=0,
            reasoning_effort="low",
            max_tokens=3000,
        )
        prompt = f"""You are a job application assistant. A candidate is filling a LinkedIn Easy Apply form.

CURRENT FORM STEP (what the user sees):
{modal_context[:1200]}

VISIBLE FORM FIELDS (unfilled):
{json.dumps(fields, separators=(',', ':'))}

CANDIDATE CV:
{json.dumps(_lean_cv_for_prompt(cv_data), separators=(',', ':'))}

TASK: Return the correct value for each field based ONLY on the candidate's CV.
Rules:
- "years of experience" -> calculate from CV dates (be conservative)
- work authorization / eligibility -> "Yes"
- "how did you hear" -> "LinkedIn"
- salary / compensation -> leave empty ("")
- "verify/confirm/re-enter email" -> repeat the exact "email" value from the CANDIDATE CV above
- "verify/confirm/re-enter phone" -> repeat the exact "phone" value from the CANDIDATE CV above
- open-text questions -> 1-2 sentences using CV evidence only, never fabricate
- yes/no radio -> answer "yes" for standard eligibility questions
- "notice period" / "when can you start" -> "Immediately" or "2 weeks"
- if a question can't be answered from CV -> return ""

Return a JSON array ONLY, no explanation:
[{{"id": "field_id", "value": "answer"}}, ...]"""

        response = retry_groq_call(
            lambda: llm.invoke([HumanMessage(content=prompt)]),
            fallback_fn=lambda: fallback_llm.invoke([HumanMessage(content=prompt)]),
        )
        text = response.content.strip()
        if "[" in text and "]" in text:
            text = text[text.find("["):text.rfind("]") + 1]
        return json.loads(text)
    except Exception as e:
        print(f"    [WARN] LLM screening answer error: {e}")

        try:

            if "state" in locals() and state.get("job_id"):

                log_step(state["job_id"], "LinkedIn Agent", f"    [WARN] LLM screening answer error: {e}")

            elif "job_id" in locals() and job_id:

                log_step(job_id, "LinkedIn Agent", f"    [WARN] LLM screening answer error: {e}")

        except Exception:

            pass
        return []


def _apply_llm_answers(driver, answers: list):
    """Apply LLM-generated field answers to the form."""
    for item in answers:
        fid = item.get("id", "")
        val = item.get("value", "")
        if not fid or not val:
            continue
        try:
            matches = _dom_query(driver, f'[id="{fid}"]')
            el = matches[0] if matches else None
            if not el or not el.is_displayed():
                continue
            tag = el.tag_name.lower()
            itype = (el.get_attribute("type") or "").lower()

            if tag == "select":
                sel_obj = Select(el)
                try:
                    selected = sel_obj.first_selected_option
                    if selected and selected.text.strip() == val:
                        continue
                    sel_obj.select_by_visible_text(val)
                except Exception:
                    if len(sel_obj.options) > 1:
                        sel_obj.select_by_index(1)
            elif itype == "radio":
                if val.lower() in ("yes", "true", "1") and not el.is_selected():
                    driver.execute_script("arguments[0].click();", el)
            elif itype == "checkbox":
                if val.lower() in ("yes", "true", "1") and not el.is_selected():
                    driver.execute_script("arguments[0].click();", el)
            else:
                current = (el.get_attribute("value") or "").strip()
                if current != val:
                    _safe_fill(driver, el, val)
        except Exception:
            pass


# ── Graph nodes ───────────────────────────────────────────────────────────────

def node_open_job(state: LinkedInState) -> dict:
    driver = state["driver"]
    print(f"  [LinkedIn Agent] Opening: {state['job_url']}")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", f"  [LinkedIn Agent] Opening: {state['job_url']}")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", f"  [LinkedIn Agent] Opening: {state['job_url']}")

    except Exception:

        pass
    try:
        driver.get(state["job_url"])
        time.sleep(3)
    except Exception as e:
        return {"result": "failed", "error": f"Could not open URL: {e}"}
    return {}


def node_check_login(state: LinkedInState) -> dict:
    driver = state["driver"]
    try:
        login_needed = _is_login_wall(driver)
        return {"login_done": not login_needed}
    except Exception:
        return {"login_done": False}


def node_wait_login(state: LinkedInState) -> dict:
    driver = state["driver"]
    timeout = 300
    print()
    print("=" * 64)
    print("  [WARN]  LINKEDIN LOGIN REQUIRED")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "  [WARN]  LINKEDIN LOGIN REQUIRED")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "  [WARN]  LINKEDIN LOGIN REQUIRED")

    except Exception:

        pass
    print()
    print("  IMPORTANT: Use your LinkedIn EMAIL + PASSWORD to sign in.")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "  IMPORTANT: Use your LinkedIn EMAIL + PASSWORD to sign in.")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "  IMPORTANT: Use your LinkedIn EMAIL + PASSWORD to sign in.")

    except Exception:

        pass
    print("  Do NOT click 'Continue with Google' — Google blocks sign-in")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "  Do NOT click 'Continue with Google' — Google blocks sign-in")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "  Do NOT click 'Continue with Google' — Google blocks sign-in")

    except Exception:

        pass
    print("  from automated browsers and the window will close.")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "  from automated browsers and the window will close.")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "  from automated browsers and the window will close.")

    except Exception:

        pass
    print()
    print("  Steps:")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "  Steps:")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "  Steps:")

    except Exception:

        pass
    print("    1. Click 'Sign in' in the Chrome window")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "    1. Click 'Sign in' in the Chrome window")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "    1. Click 'Sign in' in the Chrome window")

    except Exception:

        pass
    print("    2. Enter your LinkedIn email and password")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "    2. Enter your LinkedIn email and password")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "    2. Enter your LinkedIn email and password")

    except Exception:

        pass
    print("    3. Complete any verification if asked")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "    3. Complete any verification if asked")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "    3. Complete any verification if asked")

    except Exception:

        pass
    print("    4. The bot will continue automatically once logged in")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "    4. The bot will continue automatically once logged in")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "    4. The bot will continue automatically once logged in")

    except Exception:

        pass
    print()
    print("  After first login your session is saved — future runs")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "  After first login your session is saved — future runs")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "  After first login your session is saved — future runs")

    except Exception:

        pass
    print("  will not need to log in again.")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "  will not need to log in again.")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "  will not need to log in again.")

    except Exception:

        pass
    print(f"  Waiting up to {timeout // 60} minutes...")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", f"  Waiting up to {timeout // 60} minutes...")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", f"  Waiting up to {timeout // 60} minutes...")

    except Exception:

        pass
    print("=" * 64)
    print()

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(2)
        try:
            from src.workflow.apply import skip_event, SkipRequested, check_browser_alive
            if skip_event.is_set():
                raise SkipRequested()
            check_browser_alive(driver)
        except ImportError:
            pass
        try:
            if not _is_login_wall(driver):
                url = driver.current_url.lower()
                if "linkedin.com" in url:
                    print("  [OK] Login complete — navigating back to the job...")

                    try:

                        if "state" in locals() and state.get("job_id"):

                            log_step(state["job_id"], "LinkedIn Agent", "  [OK] Login complete — navigating back to the job...")

                        elif "job_id" in locals() and job_id:

                            log_step(job_id, "LinkedIn Agent", "  [OK] Login complete — navigating back to the job...")

                    except Exception:

                        pass
                    time.sleep(1)
                    driver.get(state["job_url"])
                    time.sleep(3)
                    return {"login_done": True}
        except Exception as e:
            err = str(e).lower()
            # Browser was closed by user or crashed
            if any(k in err for k in ("no such window", "target window", "session deleted", "unable to connect")):
                return {"result": "failed", "error": "Browser was closed during login"}
            pass

    return {"result": "failed", "error": "Login timed out after 5 minutes"}


# Module-level so both the live apply flow (below) and the discovery-time
# LinkedIn Easy-Apply-vs-external classifier (src/discovery/linkedin_classify.py)
# use the exact same detection logic instead of two copies that can drift apart.
EASY_APPLY_SELECTORS = [
    # LinkedIn's current "SDUI" job page (rolled out ~Aug 2026) dropped the literal
    # "Easy Apply" text entirely — the top-card control is an <a> that just says
    # "Apply", distinguished only by this aria-label and by linking straight to
    # /jobs/view/<id>/apply/?openSDUIApplyFlow=true instead of an outbound redirect.
    "a[aria-label='LinkedIn Apply to this job']",
    "a[aria-label*='LinkedIn Apply' i]",
    "a[href*='openSDUIApplyFlow']",
    # Pre-SDUI LinkedIn rendered the Easy Apply control as an <a>, identified by its
    # aria-label containing the literal "Easy Apply" text (class names are
    # hashed/obfuscated per load) — kept as fallback in case of A/B rollback.
    "a[aria-label*='Easy Apply' i]",
    "xpath://a[contains(@aria-label, 'Easy Apply')]",
    "xpath://span[contains(text(), 'Easy Apply')]/ancestor::a",
    # Older/alternate LinkedIn layouts render it as a <button> — kept as fallbacks.
    "button.jobs-apply-button",
    "button[aria-label*='Easy Apply' i]",
    ".jobs-apply-button--top-card",
    ".jobs-unified-top-card__content button.artdeco-button--primary",
    "button[data-control-name='jobdetails_topcard_inapply']",
    "xpath://button[contains(., 'Easy Apply')]",
    "xpath://span[contains(text(), 'Easy Apply')]/ancestor::button",
    "xpath://button[contains(@aria-label, 'Easy Apply')]",
    "xpath://span[contains(text(), 'Easy Apply')]"
]

EXTERNAL_APPLY_SELECTORS = [
    # Current SDUI layout: the outbound control's aria-label says this exact
    # phrase, and its href is always a linkedin.com/safety/go/?url=... redirect
    # wrapper around the employer's real ATS link rather than an in-app apply flow.
    "a[aria-label='Apply on company website']",
    "a[aria-label*='company website' i]",
    "a[href*='linkedin.com/safety/go/']",
    "a.jobs-apply-button",
    "a[data-control-name='jobdetails_topcard_inapply']",
    ".jobs-unified-top-card__content a.artdeco-button--primary",
    "xpath://div[contains(@class, 'jobs-unified-top-card')]//a[contains(., 'Apply')]",
    "xpath://div[contains(@class, 'jobs-unified-top-card')]//span[contains(text(), 'Apply')]/ancestor::a",
    "xpath://div[contains(@class, 'jobs-unified-top-card')]//span[contains(text(), 'Apply')]/ancestor::button",
    # Broad generic fallbacks — kept last since, on the current SDUI layout, both
    # Easy Apply and external buttons say plain "Apply" with no "Easy" in their
    # aria-label, so these alone can't tell the two apart (that's what broke
    # classification: this list used to be checked and always won). Only reached
    # if every EASY_APPLY_SELECTORS entry above already failed to match.
    "button[aria-label*='Apply' i]:not([aria-label*='Easy' i])",
    "a[aria-label*='Apply' i]:not([aria-label*='Easy' i])"
]


def node_click_easy_apply(state: LinkedInState) -> dict:
    driver = state["driver"]
    print("  [LinkedIn Agent] Running node_click_easy_apply...")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "  [LinkedIn Agent] Running node_click_easy_apply...")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "  [LinkedIn Agent] Running node_click_easy_apply...")

    except Exception:

        pass

    clicked = _click(driver, EASY_APPLY_SELECTORS, "Easy Apply button", timeout=10)

    if not clicked:
        # Check if it's a regular (non-Easy-Apply) external link
        el = _find(driver, EXTERNAL_APPLY_SELECTORS, timeout=4)
        if el:
            print("  -> External apply button found (not Easy Apply) — clicking it to open portal!")

            try:

                if "state" in locals() and state.get("job_id"):

                    log_step(state["job_id"], "LinkedIn Agent", "  -> External apply button found (not Easy Apply) — clicking it to open portal!")

                elif "job_id" in locals() and job_id:

                    log_step(job_id, "LinkedIn Agent", "  -> External apply button found (not Easy Apply) — clicking it to open portal!")

            except Exception:

                pass
            # Snapshot open tabs BEFORE the click — the external apply button typically
            # opens the company's site in a new tab, and this snapshot is the only way
            # to later tell which handle is the new one.
            handles_before_click = driver.window_handles
            clicked_ext = _click(driver, EXTERNAL_APPLY_SELECTORS, "External Apply button", timeout=5)
            if clicked_ext:
                time.sleep(2)
                # Invoke the external form handler to switch tabs, click Apply/Get Started, and fill
                job_id = state.get("job_id")
                if job_id:
                    log_step(job_id, "LinkedIn Agent", "Invoking LinkedInExternalApplier to handle external form...")
                ext_agent = LinkedInExternalApplier(
                    driver=driver,
                    cv_data=state["cv_data"],
                    cover_letter=state["cover_letter"],
                    cv_filepath=state["cv_filepath"],
                    original_handles=handles_before_click,
                )
                ext_agent.run(job_id=job_id)
                return {"result": "handled_external"}

        # Check if already applied
        try:
            body = driver.find_element(By.TAG_NAME, "body").text.lower()
            if "applied" in body and "you applied" in body:
                print("  -> Already applied to this job")

                try:

                    if "state" in locals() and state.get("job_id"):

                        log_step(state["job_id"], "LinkedIn Agent", "  -> Already applied to this job")

                    elif "job_id" in locals() and job_id:

                        log_step(job_id, "LinkedIn Agent", "  -> Already applied to this job")

                except Exception:

                    pass
                return {"result": "already_applied"}
        except Exception:
            pass

        return {"result": "failed", "error": "Easy Apply button not found on this job page"}

    time.sleep(2)

    # Verify modal opened
    modal = _find(driver, MODAL_SELECTORS, timeout=6)
    if not modal:
        return {"result": "failed", "error": "Easy Apply modal did not open after clicking"}

    print("  [OK] Easy Apply modal opened")


    try:


        if "state" in locals() and state.get("job_id"):


            log_step(state["job_id"], "LinkedIn Agent", "  [OK] Easy Apply modal opened")


        elif "job_id" in locals() and job_id:


            log_step(job_id, "LinkedIn Agent", "  [OK] Easy Apply modal opened")


    except Exception:


        pass
    return {}


def node_fill_step(state: LinkedInState) -> dict:
    """
    Fill the current modal step using deterministic selectors + LLM for unknowns.
    """
    driver = state["driver"]
    cv = state["cv_data"]

    print(f"  -> Filling modal step {state['step_count'] + 1}")


    try:


        if "state" in locals() and state.get("job_id"):


            log_step(state["job_id"], "LinkedIn Agent", f"  -> Filling modal step {state['step_count'] + 1}")


        elif "job_id" in locals() and job_id:


            log_step(job_id, "LinkedIn Agent", f"  -> Filling modal step {state['step_count'] + 1}")


    except Exception:


        pass

    # ── 1. Contact info (deterministic, force-overwrite LinkedIn prefill) ─────
    full_name = cv.get("full_name") or cv.get("name") or ""
    first_name = ""
    last_name = ""
    if full_name:
        parts = full_name.split()
        if len(parts) > 1:
            first_name = parts[0]
            last_name = " ".join(parts[1:])
        else:
            first_name = full_name

    email = cv.get("email", "")
    phone = cv.get("phone", "")
    location = cv.get("location", "")

    if first_name:
        _fill_any(driver, [
            "input[id*='firstName' i]",
            "input[name*='firstName' i]",
            "input[placeholder*='First Name' i]",
        ], first_name, skip_if_filled=False)

    if last_name:
        _fill_any(driver, [
            "input[id*='lastName' i]",
            "input[name*='lastName' i]",
            "input[placeholder*='Last Name' i]",
        ], last_name, skip_if_filled=False)

    if email:
        _fill_any(driver, [
            "input[id*='email' i]",
            "input[name*='email' i]",
            "input[placeholder*='Email' i]",
        ], email, skip_if_filled=False)

    if phone:
        _fill_any(driver, [
            "input[id*='phoneNumber-nationalNumber' i]",
            "input[id*='phoneNumber' i]",
            "input[name*='phoneNumber' i]",
            "input[type='tel']",
        ], phone, skip_if_filled=False)

    if location:
        _fill_any(driver, [
            "input[id*='city' i]",
            "input[id*='location' i]",
            "input[placeholder*='city' i]",
            "input[placeholder*='location' i]",
        ], location, skip_if_filled=False)

    # ── 2. Resume upload ───────────────────────────────────────────────────────
    cv_filepath = state.get("cv_filepath", "")
    if cv_filepath and os.path.exists(cv_filepath):
        try:
            file_inputs = _dom_query(driver, "input[type='file']")
            for fi in file_inputs:
                try:
                    fi.send_keys(os.path.abspath(cv_filepath))
                    print("    [OK] Uploaded resume")

                    try:

                        if "state" in locals() and state.get("job_id"):

                            log_step(state["job_id"], "LinkedIn Agent", "    [OK] Uploaded resume")

                        elif "job_id" in locals() and job_id:

                            log_step(job_id, "LinkedIn Agent", "    [OK] Uploaded resume")

                    except Exception:

                        pass
                    time.sleep(1.5)
                    break
                except Exception:
                    pass
        except Exception:
            pass

    # ── 3. Cover letter (deterministic) ───────────────────────────────────────
    cover_letter = state.get("cover_letter", "")
    if cover_letter:
        _fill_any(driver, [
            "textarea[id*='cover-letter' i]",
            "textarea[id*='coverLetter' i]",
            "textarea[name*='cover' i]",
            "textarea[aria-label*='cover letter' i]",
        ], cover_letter)

    # ── 4. LLM screening question answering ───────────────────────────────────
    context = _modal_text(driver)
    all_fields = _modal_fields(driver)
    fields_to_fill = [
        f for f in all_fields
        if f["type"] not in ("file", "hidden", "submit", "button")
    ]

    if fields_to_fill:
        llm_answers = _llm_answer_fields(cv, fields_to_fill, context)
        if llm_answers:
            _apply_llm_answers(driver, llm_answers)
            print(f"    [OK] LLM answered/verified {len(llm_answers)} field(s)")

            try:

                if "state" in locals() and state.get("job_id"):

                    log_step(state["job_id"], "LinkedIn Agent", f"    [OK] LLM answered/verified {len(llm_answers)} field(s)")

                elif "job_id" in locals() and job_id:

                    log_step(job_id, "LinkedIn Agent", f"    [OK] LLM answered/verified {len(llm_answers)} field(s)")

            except Exception:

                pass

    # ── 5. Fallback: radio Yes/No groups ──────────────────────────────────────
    try:
        radios = _dom_query(driver, "input[type='radio']")
        for radio in radios:
            try:
                rid = radio.get_attribute("id") or ""
                val = (radio.get_attribute("value") or "").lower()
                label_text = ""
                if rid:
                    lbls = _dom_query(driver, f'label[for="{rid}"]')
                    if lbls:
                        label_text = lbls[0].text.lower()
                if ("yes" in label_text or val == "yes") and not radio.is_selected():
                    radio.click()
            except Exception:
                pass
    except Exception:
        pass

    time.sleep(0.5)
    return {}


def node_advance(state: LinkedInState) -> dict:
    """Click the Next / Continue button to move to the next modal step."""
    driver = state["driver"]
    text_before = _modal_text(driver)

    next_selectors = [
        "button[aria-label*='Continue to next step' i]",
        "button[aria-label*='next step' i]",
        "button[aria-label='Review your application']",
        "button[data-easy-apply-next-button]",
        "xpath://button[contains(., 'Next')]",
        "xpath://button[contains(span/text(), 'Next')]",
        "xpath://button[contains(., 'Review')]",
        "xpath://button[contains(span/text(), 'Review')]"
    ]

    clicked = _click(driver, next_selectors, "Next step", timeout=5)

    if not clicked:
        # Fallback: shadow-DOM-aware text match — the modal's Next/Review button
        # lives inside LinkedIn's Easy Apply web component (shadow root), where
        # plain XPath text-matching selectors above can't reach.
        clicked = _click_shadow_by_text(
            driver, "button",
            ["continue to next step", "next", "review your application", "review", "submit application"],
            "Next/Review button"
        )

    if not clicked:
        # Fallback: click the primary button in the modal footer
        try:
            modal_btns = _dom_query(driver, ".jobs-easy-apply-modal footer button, div[role='dialog'] footer button, footer button")
            for btn in reversed(modal_btns):
                txt = (btn.text or "").strip().lower()
                aria = (btn.get_attribute("aria-label") or "").lower()
                if any(k in txt or k in aria for k in ("next", "continue", "review")):
                    driver.execute_script("arguments[0].click();", btn)
                    print("    [OK] Clicked footer Next button (fallback)")

                    try:

                        if "state" in locals() and state.get("job_id"):

                            log_step(state["job_id"], "LinkedIn Agent", "    [OK] Clicked footer Next button (fallback)")

                        elif "job_id" in locals() and job_id:

                            log_step(job_id, "LinkedIn Agent", "    [OK] Clicked footer Next button (fallback)")

                    except Exception:

                        pass
                    time.sleep(2)
                    clicked = True
                    break
        except Exception:
            pass

    # Detect a stuck step: clicked something, but the modal content didn't change
    # at all (seen with some third-party ATS embeds, e.g. Greenhouse, where the
    # Next click is silently ignored). Fail fast instead of burning all max_steps.
    stuck_count = state.get("stuck_count", 0)
    if clicked and text_before and _modal_text(driver) == text_before:
        stuck_count += 1
    else:
        stuck_count = 0

    if stuck_count >= 2:
        return {
            "step_count": state["step_count"] + 1,
            "stuck_count": stuck_count,
            "result": "failed",
            "error": "Next button click had no effect after repeated attempts — this application may use a "
                     "third-party form that needs manual completion.",
        }

    # Increment step_count by 1 to avoid infinite loop locks
    return {"step_count": state["step_count"] + 1, "stuck_count": stuck_count}


def node_review(state: LinkedInState) -> dict:
    """Reached the LinkedIn Review page — pause for human-in-the-loop."""
    print()
    print("=" * 64)
    print("  [OK]  LINKEDIN APPLICATION FULLY FILLED")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "  [OK]  LINKEDIN APPLICATION FULLY FILLED")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "  [OK]  LINKEDIN APPLICATION FULLY FILLED")

    except Exception:

        pass
    print("  Review the pre-filled form in the Chrome window.")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "  Review the pre-filled form in the Chrome window.")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "  Review the pre-filled form in the Chrome window.")

    except Exception:

        pass
    print("  Make any edits you need, then:")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "  Make any edits you need, then:")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "  Make any edits you need, then:")

    except Exception:

        pass
    print("    1. Click 'Submit application' in LinkedIn")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "    1. Click 'Submit application' in LinkedIn")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "    1. Click 'Submit application' in LinkedIn")

    except Exception:

        pass
    print("    2. Click 'Submit Done' in the Job Bot dashboard")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", "    2. Click 'Submit Done' in the Job Bot dashboard")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", "    2. Click 'Submit Done' in the Job Bot dashboard")

    except Exception:

        pass
    print("=" * 64)
    print()
    return {"result": "review_reached"}


def node_handle_error(state: LinkedInState) -> dict:
    err = state.get("error") or state.get("result") or "unknown error"
    result = state.get("result") or "failed"
    print(f"  [ERR] LinkedIn agent stopped: {err}")

    try:

        if "state" in locals() and state.get("job_id"):

            log_step(state["job_id"], "LinkedIn Agent", f"  [ERR] LinkedIn agent stopped: {err}")

        elif "job_id" in locals() and job_id:

            log_step(job_id, "LinkedIn Agent", f"  [ERR] LinkedIn agent stopped: {err}")

    except Exception:

        pass
    if result not in ("external", "handled_external", "already_applied"):
        print("  Leaving browser open — you can apply manually.")

        try:

            if "state" in locals() and state.get("job_id"):

                log_step(state["job_id"], "LinkedIn Agent", "  Leaving browser open — you can apply manually.")

            elif "job_id" in locals() and job_id:

                log_step(job_id, "LinkedIn Agent", "  Leaving browser open — you can apply manually.")

        except Exception:

            pass
    return {"result": result or "failed"}


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_after_check_login(state: LinkedInState) -> str:
    if state.get("result") in ("failed", "external", "handled_external", "already_applied"):
        return "handle_error"
    return "wait_login" if not state.get("login_done") else "click_easy_apply"


def _route_after_wait_login(state: LinkedInState) -> str:
    return "handle_error" if state.get("result") == "failed" else "click_easy_apply"


def _route_after_click_apply(state: LinkedInState) -> str:
    if state.get("result") in ("failed", "external", "handled_external", "already_applied"):
        return "handle_error"
    return "fill_step"


def _route_after_fill(state: LinkedInState) -> str:
    driver = state["driver"]
    if _is_review_page(driver):
        return "review"
    if state["step_count"] >= state["max_steps"]:
        return "handle_error"
    return "advance"


def _route_after_advance(state: LinkedInState) -> str:
    driver = state["driver"]
    if state.get("result") == "failed":
        return "handle_error"
    if _is_review_page(driver):
        return "review"
    if state["step_count"] >= state["max_steps"]:
        return "handle_error"
    return "fill_step"


# ── Build and compile graph ───────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(LinkedInState)

    g.add_node("open_job", node_open_job)
    g.add_node("check_login", node_check_login)
    g.add_node("wait_login", node_wait_login)
    g.add_node("click_easy_apply", node_click_easy_apply)
    g.add_node("fill_step", node_fill_step)
    g.add_node("advance", node_advance)
    g.add_node("review", node_review)
    g.add_node("handle_error", node_handle_error)

    g.set_entry_point("open_job")
    g.add_edge("open_job", "check_login")

    g.add_conditional_edges("check_login", _route_after_check_login, {
        "wait_login": "wait_login",
        "click_easy_apply": "click_easy_apply",
        "handle_error": "handle_error",
    })
    g.add_conditional_edges("wait_login", _route_after_wait_login, {
        "click_easy_apply": "click_easy_apply",
        "handle_error": "handle_error",
    })
    g.add_conditional_edges("click_easy_apply", _route_after_click_apply, {
        "fill_step": "fill_step",
        "handle_error": "handle_error",
    })
    g.add_conditional_edges("fill_step", _route_after_fill, {
        "advance": "advance",
        "review": "review",
        "handle_error": "handle_error",
    })
    g.add_conditional_edges("advance", _route_after_advance, {
        "fill_step": "fill_step",
        "review": "review",
        "handle_error": "handle_error",
    })

    g.add_edge("review", END)
    g.add_edge("handle_error", END)

    return g.compile()


_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run_linkedin_agent(
    driver,
    job_url: str,
    cv_data: dict,
    cover_letter: str,
    cv_filepath: str,
    job_id: str = None,
) -> str:
    """
    Run the LangGraph LinkedIn Easy Apply agent.

    Returns:
        'review_reached'  — form filled, stopped at Review page for HITL
        'external'        — job uses external apply link (not Easy Apply)
        'already_applied' — already applied to this job
        'failed'          — could not complete (browser left open for manual apply)
    """
    initial: LinkedInState = {
        "driver": driver,
        "job_url": job_url,
        "cv_data": cv_data,
        "cover_letter": cover_letter,
        "cv_filepath": cv_filepath,
        "job_id": job_id,
        "step_count": 0,
        "max_steps": 15,
        "login_done": False,
        "result": "",
        "error": "",
        "stuck_count": 0,
    }
    final = _graph.invoke(initial)
    return final.get("result", "failed")


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


class LinkedInExternalApplier:
    """
    Handles a LinkedIn job whose "Apply" button redirects to an external company
    site (not LinkedIn's own Easy Apply modal, which node_click_easy_apply /
    node_fill_step handle separately and which this class never touches). Fully
    self-contained — no longer wraps SequentialApplier from apply.py.

    Hardened this way after being diagnosed live against real jobs that got stuck
    under the previous (SequentialApplier-delegating) implementation:
      - waits for slow-hydrating pages to actually render before filling
      - clicks through an intermediate landing page in front of the real form
        (confirmed on a Dice/Ceipal and a Randstad job)
      - switches into an <iframe> if the real form is embedded in one
        (confirmed on a Fastly/Greenhouse job)
      - detects radio buttons/dropdowns with no name/id at all, grouped by their
        shared container instead (confirmed on a Ladders job)
      - recognizes non-"submit"-worded final buttons (e.g. Ladders' "Apply to
        Job") so it stops for human review instead of either getting stuck or
        risking an auto-click on a true submit button
      - uploads the system-generated, per-job cover letter (never any other file)
    """

    def __init__(self, driver, cv_data: dict, cover_letter: str = "", cv_filepath: str = None, original_handles=None):
        self.driver = driver
        self.cv_data = cv_data
        self.cover_letter = cover_letter
        self.cv_filepath = cv_filepath
        self.job_id = None
        # Maps a synthetic id (assigned while scanning) back to a stamped marker
        # attribute for custom ARIA widgets, since they have no reliable name/id
        # the way native inputs do — populated fresh on every _collect_unfilled_fields() call.
        self._custom_widget_map = {}
        # Must be the tab list captured BEFORE the external-apply button was clicked —
        # capturing it here (in __init__, called after the click) would already include
        # the new tab, making the "before vs after" diff below always find nothing and
        # falsely report "no new tab", leaving the driver stuck on the old tab.
        self.original_handles = original_handles if original_handles is not None else driver.window_handles

    # ------------------------------------------------------------------ logging

    def _log(self, message, level="INFO"):
        print(f"  [LinkedIn External] {message}")
        try:
            if self.job_id:
                log_step(self.job_id, "LinkedIn Agent", message, level=level)
        except Exception:
            pass

    # ------------------------------------------------------------------ low-level helpers

    def _recover_dead_window(self):
        """
        True if the driver was successfully switched to a live tab after its
        current one was closed out from under it. Confirmed live on a
        Snowflake Data Engineer job and an Amtrak job (both LinkedIn external
        redirects): an intermediate redirector tab can open the REAL
        application in a further new tab and then close itself a moment
        later — every operation on the now-dead handle then fails with "no
        such window: target window already closed", including every single
        field-fill selector, which cascaded into the browser appearing to
        just "close" mid-application. Mirrors the same fix in dice_apply.py
        (found live on a Randstad Digital job). Returns False if the current
        tab is still fine (nothing to recover) or if no tabs remain.
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
            self._log("Previous tab closed itself — switched to the new active tab.")
            return True
        except Exception:
            return False

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
                except Exception as e:
                    error_str = str(e).lower()
                    if "no such window" in error_str or "target window already closed" in error_str:
                        if self._recover_dead_window():
                            time.sleep(0.5)
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
            self._click_resilient(el)
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

    # Deliberately narrow (known CAPTCHA-vendor domains only) so this never fires
    # on an employer's own, unrelated iframe — same spirit as the size filter on
    # _search_iframes_for_form. Mirrors builtin_apply.py.
    _CAPTCHA_IFRAME_SIGNATURES = (
        ("captcha-delivery.com", "DataDome"),
        ("hcaptcha.com", "hCaptcha"),
        ("recaptcha", "reCAPTCHA"),
        ("arkoselabs.com", "Arkose/FunCaptcha"),
        ("funcaptcha.com", "Arkose/FunCaptcha"),
        ("perimeterx", "PerimeterX"),
        ("px-captcha", "PerimeterX"),
    )

    def _detect_captcha_challenge(self):
        """
        Returns a short vendor name if a known third-party CAPTCHA/bot-check
        iframe is visibly present AND large enough to be an actual blocking
        challenge, else None.

        The size filter is required — confirmed live on a Benjamin / Greenhouse
        job ("Senior Data Engineer"): it uses reCAPTCHA v3, which shows a small,
        ever-present, entirely passive "I'm not a robot" corner badge
        (256x60px) on nearly every page regardless of whether anything is
        actually being challenged — unlike DataDome's real blocking overlay
        (confirmed live on Turner & Townsend at 1036x655px), reCAPTCHA v3's
        badge is not a signal that anything is wrong. Without this filter, the
        very first version of this check paused the bot on this and, by
        extension, any other page using reCAPTCHA v3 passively — a regression
        on jobs that previously applied correctly. Same size threshold already
        used by _search_iframes_for_form for the same reason. Mirrors
        builtin_apply.py.
        """
        try:
            for frame_el in self.driver.find_elements(By.TAG_NAME, "iframe"):
                try:
                    if not frame_el.is_displayed():
                        continue
                    size = frame_el.size
                    if size.get("width", 0) < 200 or size.get("height", 0) < 200:
                        continue
                    src = (frame_el.get_attribute("src") or "").lower()
                except Exception:
                    continue
                for needle, vendor in self._CAPTCHA_IFRAME_SIGNATURES:
                    if needle in src:
                        return vendor
        except Exception:
            pass
        return None

    def wait_for_bot_challenge(self):
        """
        Wait if a third-party bot-detection/CAPTCHA challenge (DataDome, hCaptcha,
        reCAPTCHA, PerimeterX, Arkose/FunCaptcha, etc.) is blocking the real page
        content — confirmed live on a Turner & Townsend / SmartRecruiters job
        (reached via the same underlying pattern via LinkedIn external-apply):
        DataDome served a captcha-delivery.com iframe instead of the real
        application form, and with no detection for this the bot had no way to
        tell that apart from "this site's form uses an unrecognized layout",
        giving the misleading generic "Bot failed" message instead of saying what
        was actually going on.

        Does NOT attempt to solve or bypass the challenge — only waits for you to
        clear it (or for it to clear on its own), the same way wait_for_cloudflare()
        already does for Cloudflare's own challenge pages. Reuses
        _wait_for_human_to_advance_step() for the actual waiting rather than a new
        bespoke loop — that method already handles detecting a new tab, a closed
        browser, and (via page-signature change) the page genuinely moving on once
        you solve it, so resuming automatically afterward falls out of the
        existing, already-verified mechanism for free: once this returns,
        fill_form() just continues its normal flow against the now-real page.
        Mirrors builtin_apply.py.
        """
        # If real form content is already visible, nothing is actually blocked —
        # confirmed live on the same Benjamin / Greenhouse job: a captcha-vendor
        # iframe can coexist harmlessly on an otherwise fully-fillable page, and
        # its mere presence must never be read as "the form is blocked" on its
        # own. Only escalate when a real form genuinely can't be found at all —
        # the actual DataDome/Turner & Townsend pattern this was built for.
        if self._frame_has_form_content():
            return True

        vendor = self._detect_captcha_challenge()
        if not vendor:
            return True

        self._log(f"{vendor} security challenge detected on this page — please solve it in the browser; "
                   "the bot will resume automatically once it clears.", level="WARN")

        self._wait_for_human_to_advance_step(
            self._page_signature(),
            message=f"{vendor} security challenge detected — please solve it manually in the browser.",
        )
        return True

    def wait_for_page_ready(self, timeout=12):
        """
        Poll until at least one real interactive element (native or custom-widget)
        is visible on the page, instead of a fixed sleep followed immediately by a
        fill attempt. Slow-hydrating JS-heavy ATS forms (Taleo, Workday, some
        Greenhouse variants) can take several seconds after navigation before any
        field actually exists in the DOM — filling too early looks identical to
        "this form has nothing to fill".
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
        # Explicit "manual"/"direct" apply phrasing is checked before the generic
        # "apply" catch-all below — confirmed live on an Ovative Group / Workday
        # job that offered "Apply with LinkedIn" / "Apply with Indeed" / "Apply
        # Manually" together on one page: the generic catch-all just grabbed
        # whichever button happened to appear first in the page's HTML (an
        # SSO-integrated option the bot can never actually complete), instead
        # of the manual option that actually leads to a fillable form.
        "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'apply manually')]",
        "xpath://a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'apply manually')]",
        "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'apply directly')]",
        "xpath://a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'apply directly')]",
        "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'apply on company site')]",
        "xpath://a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'apply on company site')]",
        "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'apply on employer site')]",
        "xpath://a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'apply on employer site')]",
        "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'apply')]",
        "xpath://a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'apply')]",
        "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'get started')]",
        "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue')]",
    ]

    # A SINGLE match here isn't enough proof of a real form — confirmed live that a
    # company's own nav bar can have a language-selector <button role="combobox">
    # (a legitimate accessible dropdown pattern for a totally unrelated widget)
    # that satisfies this on its own, which would mask a real form sitting in a
    # sibling <iframe> from ever being found. A real application form always has
    # several fields together (name/email/phone/etc at minimum); an isolated nav
    # widget doesn't, so requiring a handful of matches together is what tells
    # them apart.
    _FORM_FIELD_SIGNAL_SELECTOR = (
        "input[type='email'], input[type='tel'], input[type='file'], textarea, select, "
        "input[type='text'], input:not([type]), " + CUSTOM_WIDGET_ROLE_SELECTOR
    )
    _FORM_FIELD_MIN_COUNT = 3

    # A single one of THESE fields is trusted on its own, bypassing the >=3 count
    # above entirely — confirmed live on a PayPal / Eightfold job: its first
    # wizard step ("Resume") shows nothing but one required resume upload field,
    # with name/email/phone only appearing on the next step ("Contact
    # Information"), so the >=3 count below always failed on it and the bot gave
    # up claiming it couldn't find the real form — even though it was already
    # looking right at it. None of these fields ever appear in the false-positive
    # widgets the >=3 count was built to reject (Yamaha's own nav-bar job-SEARCH
    # box was just Keyword+Location text fields; SIERTEK's quick-apply widget was
    # email+city+keyword) — a resume upload or a first/last-name/phone field is
    # never part of generic site chrome, only a genuine application. Mirrors
    # builtin_apply.py, which already had this; this file was previously missing
    # it and would escalate to "please try manually" on any single-field wizard
    # step like PayPal's.
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

    def _frame_has_form_content(self):
        """True if the CURRENT frame (whatever Selenium is focused on right now)
        looks like it contains a real application form — used both to check the
        top-level page and to test candidate iframes."""
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
        Many ATS forms (Greenhouse's embeddable job-board widget confirmed live)
        are embedded inside an <iframe> on the company's own branded careers page,
        rather than the form being served directly on the visited page. Selenium
        only ever looks inside whatever frame it's currently focused on — by
        default the top-level document — so every field and button inside such an
        iframe is completely invisible to every other method in this class until
        something explicitly switches into it.

        Checks iframes up to two levels deep (not unboundedly) — confirmed live on
        a Turner & Townsend (SmartRecruiters-style) job: its real application form
        is an iframe nested INSIDE another iframe, so a top-level-only search always
        came up empty, leaving the bot filling nothing (or grabbing an unrelated
        field from the outer page) before giving up. The depth cap, plus the same
        size filter used at every level below, keeps this from wandering
        arbitrarily deep into unrelated embedded widgets (ads, chat widgets,
        trackers).
        """
        if self._frame_has_form_content():
            return True  # top-level page already has real form content — nothing to do

        if self._search_iframes_for_form(max_depth=2):
            return True

        self._log("No form content found on the page or in any embedded iframe.", level="WARN")
        return False

    def _search_iframes_for_form(self, max_depth=2):
        """
        Recursively searches iframes in the CURRENT frame for real form content, up
        to max_depth levels of nesting. Uses switch_to.parent_frame() (not
        default_content()) to back out of a frame that didn't pan out —
        default_content() always jumps to the outermost document, which would
        incorrectly skip the current level's remaining sibling iframes when
        searching two levels deep. On success, deliberately leaves Selenium's focus
        inside the matching frame for the caller to keep using. Mirrors
        builtin_apply.py.
        """
        if max_depth <= 0:
            return False

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
                try:
                    self.driver.switch_to.parent_frame()
                except Exception:
                    self.driver.switch_to.default_content()
                continue

            if self._frame_has_form_content():
                self._log(f"Switched into an <iframe> containing the real application form (id='{frame_id}')")
                return True

            if self._search_iframes_for_form(max_depth=max_depth - 1):
                return True

            try:
                self.driver.switch_to.parent_frame()
            except Exception:
                self.driver.switch_to.default_content()

        return False

    def _click_landing_page_apply_if_needed(self):
        """
        If the page already shows real form fields, does nothing UNLESS a
        distinct Apply-labeled button/link is also present on the same page —
        confirmed live on a SIERTEK LTD / LinkedIn external-apply job: an
        email + city + keyword "quick apply/search" widget sat right next to
        the real Apply button and, on its own, satisfied the >=3-fields
        "form already open" check below, so the bot filled those unrelated
        boxes and never clicked the real Apply button at all. Otherwise looks
        for an Apply/Easy Apply/Continue/Get Started button and clicks it, handling
        a possible resulting new tab — mirrors the same tab-detection approach used
        everywhere else in this file (snapshot handles before the click, not after).

        Uses _frame_has_form_content() (identity-field-first, keyword-filtered
        count fallback) rather than a single generic-field match — confirmed live
        on a BNSF job (runs Phenom People's career-site platform): a persistent
        site-wide AI-chatbot search box ("Ask anything") is visible on every page,
        including the job posting, and satisfied the old one-match check, so the
        real "Apply Now" link sitting right next to it never got clicked at all.
        """
        self._landing_apply_click_happened = False
        already_has_form = False
        try:
            already_has_form = self._frame_has_form_content()
        except Exception:
            pass

        handles_before = self.driver.window_handles

        if already_has_form:
            # Only a single, near-instant pass here — not the full retrying
            # search used below when no form exists yet at all. The button is
            # either already rendered right now (the page has clearly
            # finished loading enough to show the other fields) or it
            # genuinely isn't there — no reason to spend several seconds
            # searching on every ordinary job that lands directly on its real
            # form with no separate Apply button at all.
            clicked = self._click(
                self._LANDING_PAGE_APPLY_SELECTORS,
                "landing-page Apply button (alongside already-visible fields)",
                timeout=0.3,
            )
            if not clicked:
                return True  # no distinct Apply button — the visible fields really are the form
        else:
            self._log("No form fields visible yet — looking for a landing-page Apply/Continue button...")
            clicked = self._click(self._LANDING_PAGE_APPLY_SELECTORS, "landing-page Apply button", timeout=10)
            if not clicked:
                self._log("No landing-page Apply button found either — proceeding as-is.", level="WARN")
                return False

        self._landing_apply_click_happened = True
        time.sleep(2.5)
        new_handles = self.driver.window_handles
        if len(new_handles) > len(handles_before):
            new_tab = [h for h in new_handles if h not in handles_before][-1]
            self.driver.switch_to.window(new_tab)
            self._log("Switched to new tab/window after landing-page Apply click.")

        # Safety net for the same tab-closes-itself pattern _recover_dead_window()
        # handles below — confirmed live on an Amtrak job: this exact click can
        # open an intermediate redirector tab that opens the real form in a
        # further new tab and then closes itself a moment later.
        self._recover_dead_window()

        self.wait_for_cloudflare()
        self.wait_for_page_ready(timeout=10)
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

    def _file_input_already_has_file(self, el):
        """
        True if this file input already holds a file — checked via several
        signals, not just the data-bot-uploaded marker above. Confirmed live on
        a ReUp Education job reached via a LinkedIn-external-apply redirect
        (Rippling ATS, same underlying pattern applies here): after the resume
        upload succeeded, Rippling replaced BOTH file inputs' underlying DOM
        nodes (not just the one that changed) — wiping the marker and resetting
        files.length to 0 on the fresh node, even though the page's own rendered
        text for that slot still correctly showed the uploaded filename and a
        "Remove" action. The cover-letter upload's "grab any unfilled file
        input" fallback then saw that fresh, empty-looking node sitting in the
        resume's position and uploaded straight into it, overwriting the
        resume. Reading the page's own rendered text as an extra signal
        survives exactly this kind of re-render, since it reflects the ATS's
        real internal state rather than a DOM attribute or a raw file input
        that can both be wiped by a node swap.

        The filename-extension + remove/replace/delete combination (rather than
        either alone) is deliberate: an EMPTY dropzone's own placeholder text
        commonly mentions accepted extensions too (confirmed live on the same
        Rippling job: "drop or select (.doc / .docx / .pdf)") — matching only on
        an extension being present would treat every untouched field as already
        filled. Requiring a remove/replace/delete action word alongside it is
        what actually distinguishes "a file is attached" from "these are the
        accepted types".
        """
        try:
            if el.get_attribute("data-bot-uploaded"):
                return True
            files_len = self.driver.execute_script("return arguments[0].files ? arguments[0].files.length : 0;", el)
            if files_len:
                return True
            context = self._nearby_container_text(el)
            if re.search(r'\b[1-9]\d*\s+files?\s+selected\b', context):
                return True
            if "uploaded successfully" in context or "successfully uploaded" in context:
                return True
            has_filename_ext = re.search(r'\.(pdf|docx?|txt|rtf)\b', context)
            has_remove_action = re.search(r'\b(remove|replace|delete)\b', context)
            if has_filename_ext and has_remove_action:
                return True
        except Exception:
            pass
        return False

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
                return [el for el in elements if not self._file_input_already_has_file(el)]

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

            # Last resort: grab any remaining unfilled file input — but never
            # one contextually labeled as the OTHER kind of document, in
            # either direction. Confirmed live on an OpenAI job: this used to
            # only protect the resume-upload direction, so a cover-letter
            # upload that found nothing labeled "cover" fell through and
            # landed on the visually-obvious Resume box instead, since the
            # real resume upload had already gone to a different, less
            # obvious input matched by a generic id*='resume' selector above.
            all_inputs = _unfilled(self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']"))
            exclude_kw = "resume" if kind == "cover_letter" else "cover"
            all_inputs = [el for el in all_inputs if exclude_kw not in self._nearby_container_text(el)]
            if all_inputs:
                all_inputs[0].send_keys(os.path.abspath(filepath))
                self._log(f"Uploaded {kind} file via first unfilled file input")
                self._mark_uploaded(all_inputs[0])
                time.sleep(1.5)
                return True

            # Nothing matched at all, and not for lack of trying — confirmed
            # live on a New York Power Authority job: a required Resume field
            # sat empty because no input[type='file'] existed anywhere on the
            # page reachable by any strategy above, and this returned False
            # completely silently every single pass, giving no indication
            # anything was even attempted.
            if not self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']"):
                self._log(f"No file input found on the page for {kind} upload — this form may need it uploaded manually.", level="WARN")
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

        # Confirmed live on a Snowflake Data Engineer job and an Amtrak job: if
        # the current tab closed itself out from under us (an intermediate
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
            "input[placeholder*='full name' i]",
            # Ashby's own single-field "Full Name" convention (id/name literally
            # "_systemfield_name") — confirmed live on a Notion job reached via
            # LinkedIn's external-apply path (same underlying pattern applies
            # here): it matches none of the selectors above, so it was never
            # filled deterministically and depended entirely on the LLM
            # fallback, which silently missed it during a Groq rate-limit
            # fallback to a weaker model.
            "input[name*='_systemfield_name' i]", "input[id*='_systemfield_name' i]",
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
                # name/aria-label of its own (e.g. a Gender/Race <select> where the
                # word "Gender" lives in a sibling <div>, not on the <select> at all).
                context = aria + " " + name + " " + text_content
                if not any(k in context for k in ("sponsor", "authorized", "legally", "permit", "relocat", "gender")):
                    context += " " + self._field_label(sel).lower()

                is_sponsorship = any(k in context for k in ("sponsor", "visa", "sponsorship"))
                is_auth = any(k in context for k in ("authorized", "legally", "permit"))
                is_relocate = "relocat" in context
                is_gender = "gender" in context
                is_veteran = "veteran" in context

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
                    # Fixed answer: always select "Male" when that exact option exists
                    # (per explicit instruction — applied consistently across every
                    # application, same as relocation). Falls back to a "decline to
                    # answer" style option only when no Male option exists at all.
                    matched = False
                    for opt in sel_obj.options:
                        opt_text = (opt.text or "").strip().lower()
                        if opt_text in ("male", "m"):
                            sel_obj.select_by_visible_text(opt.text)
                            self._log(f"Gender dropdown auto-filled: {opt.text}")
                            matched = True
                            break
                    if not matched:
                        for opt in sel_obj.options:
                            opt_text = (opt.text or "").lower()
                            if any(k in opt_text for k in DECLINE_TO_ANSWER_KEYWORDS):
                                sel_obj.select_by_visible_text(opt.text)
                                self._log(f"Gender/self-ID dropdown auto-filled: {opt.text}")
                                break
                elif is_veteran:
                    # Fixed answer: always select the option meaning "not a protected
                    # veteran" (matches both "I am not a Veteran" and "I am not a
                    # protected veteran" style wording) — never an option that
                    # affirmatively claims veteran status.
                    for opt in sel_obj.options:
                        opt_text = (opt.text or "").strip().lower()
                        if "veteran" in opt_text and "not" in opt_text:
                            sel_obj.select_by_visible_text(opt.text)
                            self._log(f"Veteran status dropdown auto-filled: {opt.text}")
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
        (a Ladders Gender/Race <select> is a confirmed real example).
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
                elif any(k in context for k in ("sponsor", "visa")):
                    for g in group:
                        label = self._radio_label(g)
                        if label in ("no", "false", "n") and not g.is_selected():
                            self.driver.execute_script("arguments[0].click();", g)
                            self._log(f"Sponsorship radio auto-filled ('{name}'): {label}")
                            break
                elif "gender" in context or "sex" in context.split():
                    # Fixed answer: always select "Male" when that exact option exists;
                    # falls back to a "decline to answer" style option only when no
                    # Male option exists at all.
                    matched = False
                    for g in group:
                        label = self._radio_label(g)
                        if label in ("male", "m") and not g.is_selected():
                            self.driver.execute_script("arguments[0].click();", g)
                            self._log(f"Gender radio auto-filled ('{name}'): {label}")
                            matched = True
                            break
                    if not matched:
                        for g in group:
                            label = self._radio_label(g)
                            if any(k in label for k in DECLINE_TO_ANSWER_KEYWORDS) and not g.is_selected():
                                self.driver.execute_script("arguments[0].click();", g)
                                self._log(f"Gender/self-ID radio auto-filled ('{name}'): {label}")
                                break
                elif "veteran" in context:
                    # Fixed answer: always select the option meaning "not a protected
                    # veteran" — never one that affirmatively claims veteran status.
                    for g in group:
                        label = self._radio_label(g)
                        if "veteran" in label and "not" in label and not g.is_selected():
                            self.driver.execute_script("arguments[0].click();", g)
                            self._log(f"Veteran status radio auto-filled ('{name}'): {label}")
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
                    # Fixed answer: always select "Male" when that exact option exists;
                    # falls back to a "decline to answer" style option only when no
                    # Male option exists at all.
                    male_opt = next((o for o in options if o.strip().lower() in ("male", "m")), None)
                    if male_opt:
                        if self._click_unnamed_radio_option(key, male_opt):
                            self._log(f"Gender radio auto-filled ('{key}'): {male_opt}")
                    else:
                        for opt in options:
                            if any(k in opt.lower() for k in DECLINE_TO_ANSWER_KEYWORDS):
                                if self._click_unnamed_radio_option(key, opt):
                                    self._log(f"Gender/self-ID radio auto-filled ('{key}'): {opt}")
                                break
                elif "veteran" in question:
                    # Fixed answer: always select the option meaning "not a protected
                    # veteran" — never one that affirmatively claims veteran status.
                    for opt in options:
                        opt_lower = opt.lower()
                        if "veteran" in opt_lower and "not" in opt_lower:
                            if self._click_unnamed_radio_option(key, opt):
                                self._log(f"Veteran status radio auto-filled ('{key}'): {opt}")
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

    def _fill_known_custom_dropdowns(self):
        """
        Rule-based auto-fill for the same fixed-answer questions _fill_known_dropdowns()
        and _fill_known_radio_groups() already handle deterministically, but for custom
        ARIA-widget dropdowns/radio-groups (role='combobox'/'listbox'/'radiogroup') that
        aren't native <select>/<input type='radio']> elements — confirmed live on a PwC/
        Workday job ("Managed Services - Data Engineer") reached via the external-apply
        path, where gender/ethnicity/veteran were exactly this widget type and previously
        depended entirely on the LLM correctly guessing them instead of a fixed,
        always-correct answer. Runs BEFORE _collect_unfilled_fields() so any widget
        answered here is already filled and never gets sent to the LLM at all.
        """
        try:
            for el in self.driver.find_elements(By.CSS_SELECTOR, CUSTOM_WIDGET_ROLE_SELECTOR):
                try:
                    if not el.is_displayed():
                        continue
                    role = (el.get_attribute("role") or "").lower() or "combobox"
                    if role == "switch":
                        continue
                    if not self._is_custom_widget_unfilled(el, role):
                        continue

                    label = (self._custom_widget_question_text(el)[:200] or self._field_label(el) or "").lower()
                    is_sponsorship = any(k in label for k in ("sponsor", "visa"))
                    is_veteran = "veteran" in label
                    is_gender = "gender" in label or "sex" in label.split()
                    if not (is_sponsorship or is_veteran or is_gender):
                        continue

                    options = self._read_custom_widget_options(el, role)
                    if not options:
                        continue

                    target = None
                    kind_label = ""
                    if is_sponsorship:
                        kind_label = "Sponsorship"
                        target = next((o for o in options if o.strip().lower() in ("no", "false", "n")), None)
                    elif is_veteran:
                        kind_label = "Veteran status"
                        target = next((o for o in options if "veteran" in o.lower() and "not" in o.lower()), None)
                    elif is_gender:
                        kind_label = "Gender"
                        target = next((o for o in options if o.strip().lower() in ("male", "m")), None)
                        if not target:
                            kind_label = "Gender/self-ID"
                            target = next((o for o in options if any(k in o.lower() for k in DECLINE_TO_ANSWER_KEYWORDS)), None)
                    if not target:
                        continue

                    if role == "radiogroup":
                        for child in el.find_elements(By.CSS_SELECTOR, "[role='radio']"):
                            opt_label = (child.text or child.get_attribute("aria-label") or "").strip()
                            if opt_label.lower() == target.strip().lower():
                                self.driver.execute_script("arguments[0].click();", child)
                                self._log(f"Custom {kind_label} radio auto-filled: {opt_label}")
                                break
                    else:
                        # _read_custom_widget_options() just above already did its own
                        # open-read-close (Escape) cycle on this same element — clicking
                        # to reopen immediately afterward can race that close animation
                        # and silently toggle right back shut instead of opening (confirmed
                        # live on a PwC/Workday job reached via the external-apply path:
                        # this always came back empty-handed and fell through to the LLM
                        # with no error or log line at all). One retry with a settle delay
                        # in between is enough to reopen reliably.
                        opened_ok = False
                        for attempt in range(2):
                            time.sleep(0.3)
                            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                            self._click_resilient(el)
                            time.sleep(0.35)
                            opts_now = self.driver.find_elements(By.CSS_SELECTOR, "[role='option']")
                            visible_opts = [o for o in opts_now if o.is_displayed()]
                            if not visible_opts:
                                continue
                            for opt_el in visible_opts:
                                if (opt_el.text or "").strip().lower() == target.strip().lower():
                                    self.driver.execute_script("arguments[0].click();", opt_el)
                                    self._log(f"Custom {kind_label} dropdown auto-filled: {target}")
                                    opened_ok = True
                                    break
                            break
                        if not opened_ok:
                            self._log(f"Custom {kind_label} dropdown found target '{target}' but could not reopen/select it — leaving for LLM fallback.", level="WARN")
                            try:
                                el.send_keys(Keys.ESCAPE)
                            except Exception:
                                pass
                except Exception:
                    continue
        except Exception as e:
            self._log(f"Custom dropdown known-answer auto-fill error: {e}", level="WARN")

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

    def _fill_skills_field(self, cv_data):
        """
        Workday-style "Skills" fields are type-and-click autocomplete tag pickers,
        not plain text boxes — confirmed live on a PwC/Workday job reached via
        LinkedIn's external-apply path (same underlying pattern applies here): the
        generic LLM text fallback was blindly typing the entire skills list as one
        string into the search input, which never triggers Workday's real "add
        skill chip" mechanism, so nothing was actually selected even though it
        logged as filled. Drives the real interaction instead: type one skill,
        wait for a suggestion to appear, click it, clear, repeat. If no
        suggestions ever appear for a given field, it isn't a chip picker — leave
        it alone so the existing plain-text fallback can still handle a
        genuinely free-text "skills" field on some other ATS without regressing
        that case.
        """
        skills = cv_data.get("skills", [])
        if not isinstance(skills, list) or not skills:
            return
        try:
            candidates = self.driver.find_elements(
                By.CSS_SELECTOR,
                "input[id*='skill' i], input[name*='skill' i], input[aria-label*='skill' i]"
            )
            for el in candidates:
                try:
                    if not el.is_displayed() or el.get_attribute("data-bot-uploaded") or el.get_attribute("type") in ("hidden", "file", "password"):
                        continue
                except Exception:
                    continue

                added_count = 0
                for skill in skills[:10]:
                    try:
                        self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                        el.click()
                    except Exception:
                        pass
                    try:
                        self.driver.execute_script(_NATIVE_SET_VALUE_JS, el, "")
                    except Exception:
                        pass
                    try:
                        el.send_keys(str(skill))
                    except Exception:
                        break
                    time.sleep(0.6)

                    options = [o for o in self.driver.find_elements(By.CSS_SELECTOR, "[role='option']") if o.is_displayed()]
                    if not options:
                        break

                    skill_l = str(skill).strip().lower()
                    match = next((o for o in options if (o.text or "").strip().lower() == skill_l), None) or options[0]
                    self.driver.execute_script("arguments[0].click();", match)
                    added_count += 1
                    time.sleep(0.3)

                if added_count:
                    try:
                        self.driver.execute_script(_NATIVE_SET_VALUE_JS, el, "")
                    except Exception:
                        pass
                    self._mark_uploaded(el)
                    self._log(f"Skills field filled via autocomplete tag picker ({added_count} skill(s) added)")
        except Exception as e:
            self._log(f"Skills field auto-fill error: {e}", level="WARN")

    _TODAY_DATE_KEYWORDS = ("today", "current date", "signature date", "certification date", "date of application", "todays date")
    _WORK_HISTORY_DATE_EXCLUDE_KEYWORDS = ("start", "end", "graduation", "education", "employment", "experience")

    def _fill_today_date_fields(self):
        """
        Fills date fields representing "today's date" — a signature, certification,
        or "date of application" field — distinct from _fill_date_inputs() above,
        which only covers native <input type="date">. Workday (and similar ATS's)
        render dates as separate mm/dd/yyyy text or select inputs instead, which
        otherwise fall through to the generic LLM fallback; since a today's-date
        field has no corresponding CV data, the LLM's own "leave unanswerable
        fields empty" rule correctly-by-its-own-logic left it blank, causing a
        validation error (confirmed live on a PwC/Workday job reached via
        LinkedIn's external-apply path: "mm | dd" / "mm | yyyy" left empty).
        Explicitly excludes anything that looks like a work-history/education
        start/end date, which must stay CV-driven via the LLM pass, not
        overwritten with today's date.
        """
        try:
            now = time.localtime()
            month, day, year = str(now.tm_mon), str(now.tm_mday), str(now.tm_year)

            candidates = self.driver.find_elements(
                By.CSS_SELECTOR,
                "input:not([type='hidden']):not([type='file']):not([type='password']), select"
            )
            for el in candidates:
                try:
                    if not el.is_displayed() or el.get_attribute("data-bot-uploaded"):
                        continue
                    current = (el.get_attribute("value") or "").strip()
                    if current:
                        continue
                    haystack = " ".join([
                        el.get_attribute("id") or "", el.get_attribute("name") or "",
                        el.get_attribute("aria-label") or "", self._field_label(el),
                        self._nearby_container_text(el),
                    ]).lower()
                    if not any(k in haystack for k in self._TODAY_DATE_KEYWORDS):
                        continue
                    if any(k in haystack for k in self._WORK_HISTORY_DATE_EXCLUDE_KEYWORDS):
                        continue

                    if "month" in haystack or haystack.strip().endswith("mm"):
                        value = month.zfill(2)
                    elif "day" in haystack or haystack.strip().endswith("dd"):
                        value = day.zfill(2)
                    elif "year" in haystack or "yyyy" in haystack:
                        value = year
                    else:
                        value = f"{month.zfill(2)}/{day.zfill(2)}/{year}"

                    tag = el.tag_name.lower()
                    if tag == "select":
                        self._select_dropdown_by_text_variants(el, [value, value.zfill(2)])
                    else:
                        el.click()
                        el.send_keys(value)
                    self._log(f"Today's-date field filled: {value}")
                except Exception:
                    continue
        except Exception as e:
            self._log(f"Today's-date field auto-fill error: {e}", level="WARN")

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
            # No id/label/aria at all — common on no-name/no-id fields (same root
            # pattern as the unnamed radios above). The question text is usually a
            # sibling element rendered just before the field within their shared
            # parent, so check preceding siblings before giving up with a blank
            # label the LLM has nothing to go on.
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
        Best-effort peek at a custom widget's option list. Opens it, reads visible
        [role='option']/li/.option text, then closes it again via Escape so page
        state is restored. Never allowed to break the caller — any failure here
        just means this one field gets skipped, not that the whole scan aborts.

        Opening uses a real click, never a bare execute_script one — confirmed
        live on a real Greenhouse job (react-select, Take-Two Interactive)
        reached via the external-apply path: a JS `el.click()` on the widget left
        `aria-expanded` stuck on "false" and no options ever rendered, while a
        genuine WebDriver click opened it correctly every time. Workday's widgets
        tolerate a JS click; react-select apparently needs a real, trusted
        click/focus event to trigger its open handler.

        Even the real click is occasionally racy on that same Greenhouse widget —
        confirmed live it opened on some attempts and silently no-opped on others
        with no exception either way, purely a timing thing (page still settling
        after scrollIntoView) — so one retry with a fresh click is worth it before
        giving up and returning no options at all.
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

            opts = []
            for attempt in range(2):
                self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                time.sleep(0.15)
                self._click_resilient(el)
                time.sleep(0.35)
                for opt_el in self.driver.find_elements(By.CSS_SELECTOR, "[role='option'], [role='listbox'] li, [role='listbox'] div"):
                    if opt_el.is_displayed():
                        text = (opt_el.text or "").strip()
                        if text and text not in opts:
                            opts.append(text)
                    if len(opts) >= 12:
                        break
                if opts:
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

    # Confirmed live on a real Greenhouse job (Take-Two Interactive, job-boards.
    # greenhouse.io) reached via the external-apply path: unlike Workday,
    # Greenhouse's React form wraps each custom dropdown in several nested layout
    # divs that each individually render nothing but the widget's own "Select..."
    # placeholder — climbing ancestors and stopping at the first non-empty text
    # (the approach used elsewhere in this file) never reaches the real question
    # text at all, it just returns "Select..." over and over. The real label
    # ("Will you require an H-1B Visa transfer?", etc.) was found 5 DOM levels up,
    # past five "Select..."-only wrapper divs — so the climb must explicitly look
    # past generic placeholder-only text, not just past text identical to the
    # widget's own.
    _CUSTOM_WIDGET_LABEL_JS = """
        const PLACEHOLDERS = new Set(['select...','select one','select','choose...','choose one','please select','']);
        let node = arguments[0].parentElement;
        for (let i = 0; i < 25 && node; i++) {
            const raw = (node.innerText || node.textContent || '').trim();
            const norm = raw.toLowerCase();
            if (raw && raw.length < 400 && !PLACEHOLDERS.has(norm)) {
                return raw;
            }
            node = node.parentElement;
        }
        return '';
    """

    def _custom_widget_question_text(self, el):
        """Best-effort question label for a custom ARIA dropdown/radiogroup — see
        _CUSTOM_WIDGET_LABEL_JS for why this can't just reuse _nearby_container_text()."""
        try:
            return (self.driver.execute_script(self._CUSTOM_WIDGET_LABEL_JS, el) or "").lower()
        except Exception:
            return ""

    # Anti-bot "honeypot" fields are deliberately kept in the DOM (often
    # visually hidden via off-screen CSS rather than display:none, so a plain
    # is_displayed() check doesn't catch them) — a real applicant would never
    # see or fill one, but a bot that treats every "unfilled" field as
    # answerable does. Confirmed live on a FOX Careers / Workday signup form:
    # a field literally labeled "Enter website. This input is for robots
    # only, do not enter if you're human." got answered by the LLM like any
    # other text field, which is the most likely reason the site's "Create
    # Account" click afterward had no visible effect — a classic silent
    # honeypot-triggered rejection, no error shown.
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
        Extended field scan: native input/select/textarea/radio/checkbox (as
        before) PLUS custom ARIA widgets (combobox/listbox/radiogroup/switch) that
        the plain native-element scan cannot see at all — this is the fix for the
        several-blank-Continue-clicks-in-a-row failure mode seen on forms built
        from custom-styled components instead of native <select>/<input radio>.
        """
        fields = []
        self._custom_widget_map = {}
        seen_radio_groups = set()
        seen_custom = set()

        try:
            # Password fields are deliberately excluded here — a signup/account-
            # creation password must never be sent to the LLM as an "answerable"
            # field (confirmed live on a hackajob external-apply form: it wasn't
            # fabricated this time since the LLM correctly refused to guess, but
            # it's not a risk worth leaving open). See _has_unfilled_password_field()
            # / _wait_for_password_entry() for how password fields are actually handled.
            elements = self.driver.find_elements(
                By.CSS_SELECTOR,
                "input:not([type='hidden']):not([type='file']):not([type='password']), select, textarea"
            )
            for el in elements:
                try:
                    if not el.is_displayed():
                        continue
                    # Marks a field already handled deterministically (e.g. a skills
                    # autocomplete picker whose search box is intentionally left empty
                    # after each chip is added) — without this, this scan would see
                    # that empty box and hand it right back to the LLM text fallback,
                    # undoing the real chip-by-chip selections just made.
                    if el.get_attribute("data-bot-uploaded"):
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
                            # No id/name at all (e.g. a Ladders Gender/Race select) —
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
                    # NOTE: aria-haspopup="listbox" (matched by
                    # CUSTOM_WIDGET_ROLE_SELECTOR above) marks a dropdown that
                    # opens a listbox popup — it must fall through to the
                    # "combobox" default below, not be misread as a toggle
                    # switch. A prior version guessed role="switch" whenever
                    # aria-haspopup was "listbox" and no explicit role
                    # attribute was present (common on real dropdowns that
                    # skip role="combobox"), which fed the LLM ["on", "off"]
                    # instead of the widget's real options — confirmed live
                    # on a FOX Careers custom dropdown (__custom_cw0).
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
            # _scan_unnamed_radio_groups) — fixes forms where these were previously
            # invisible to this scan entirely, so "Continue" kept clicking without
            # ever answering them.
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
        When a Continue click has no visible effect, scan for validation-error
        indicators instead of just giving up silently — so the log states exactly
        which field is blocking progress.
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
            llm = ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-120b", temperature=0, reasoning_effort="low", max_tokens=3000)
            fallback_llm = ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-20b", temperature=0, reasoning_effort="low", max_tokens=3000)
            today_str = time.strftime("%m/%d/%Y")
            prompt = f"""You are a job application assistant filling out an external company application form
reached via a LinkedIn "Apply" redirect (not LinkedIn's own Easy Apply flow).

TODAY'S DATE: {today_str}

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
- gender / sex / voluntary self-identification -> pick the option meaning "Male"; only if no such
  option exists, pick one meaning "Decline to self-identify" / "Prefer not to say" instead; never
  pick "Female" or any other option.
- veteran status / protected veteran self-identification -> pick the option meaning "not a
  protected veteran" / "I am not a Veteran" (containing both "veteran" and "not"); never pick an
  option that affirmatively claims veteran status or a "decline to answer" option.
- other yes/no eligibility, background, or compliance questions with no CV evidence either way
  (e.g. "are you a current/former government employee", "do you have/can you obtain a security
  clearance", "are you subject to any non-compete/non-solicit/NDA restrictions", criminal-history
  or background-check consent questions) -> pick the option meaning "No" — these are gating
  questions that must be answered to proceed, and "No" is the standard non-disqualifying default
  when the CV gives no reason to answer otherwise.
- "verify/confirm/re-enter email" -> repeat the exact "email" value from the CANDIDATE CV above.
- open-text questions -> 1-2 sentences using CV evidence only, never fabricate.
- a field asking for "today's date", a signature date, a certification date, or "date of
  application" (NOT a work-history or education start/end date, which must still come from
  the CANDIDATE CV) -> answer with TODAY'S DATE above.
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
                        # Real click, not a bare execute_script one — see
                        # _read_custom_widget_options() for why (react-select /
                        # Greenhouse confirmed live: a JS click never opened it).
                        self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                        self._click_resilient(el)
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

    def _has_unfilled_password_field(self):
        """
        True if a visible <input type="password"> on the page still has no value —
        confirmed live on a hackajob external-apply job: LinkedIn's "Apply" redirect
        can land on a "create your account" signup form requiring a password the
        bot has no safe way to invent. Checks ALL password fields (a signup form
        commonly has both "Password" and "Confirm password"), not just the first.
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
        present) into the signup form yourself. Kept as a fallback for
        _generate_and_fill_password() below in case auto-filling doesn't stick on
        some form. Polls until every visible password field has a value,
        refreshing the SAME 180-second review-deadline the dashboard already
        shows (not a second, separate timer) each cycle, so the countdown doesn't
        expire out from under you while you're mid-typing. No hard cutoff of its
        own: if the tab/browser is closed, the driver calls here raise and unwind
        normally through the existing error handling, same as everywhere else in
        this class.
        """
        self._log("Waiting for you to create your password — please enter and confirm it, "
                   "the bot will continue automatically.", level="WARN")
        while True:
            try:
                from src.workflow.apply import skip_event, SkipRequested, check_browser_alive, BrowserClosedByUser
                if skip_event.is_set():
                    raise SkipRequested()
                check_browser_alive(self.driver)
            except ImportError:
                pass

            # _has_unfilled_password_field() fails safe to False on any read
            # error, which used to look identical to "you filled it in" —
            # the same false-success gap as _wait_for_human_to_advance_step()
            # applies here too, so it needs the same explicit liveness check
            # before trusting that reading.
            try:
                window_open = bool(self.driver.window_handles)
            except Exception:
                window_open = False
            if not window_open:
                raise BrowserClosedByUser("Browser window closed while waiting for password entry")

            if not self._has_unfilled_password_field():
                break
            time.sleep(2)
        self._log("Password entered — continuing automatically.")

    # Injected directly into the page currently open in the automated browser
    # (not just the log/dashboard) so it's impossible to miss while watching
    # that window — a fixed banner with a live countdown, requested directly
    # after testing showed the log-only "This step needs your help" message
    # is easy to not notice if you're not staring at the terminal.
    # pointer-events:none is required — without it, this full-width bar
    # pinned to the top of the viewport physically intercepts clicks on any
    # field/button that happens to sit near the top of the page underneath
    # it, blocking the exact manual fix it's asking you to make. Caught live
    # via a synthetic regression test (ElementClickIntercepted pointing at
    # this banner), not yet confirmed against a real job, but the fix is
    # unambiguous either way.
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
        still be visible before resuming — confirmed live on a Workday-based
        application (GM Motorsports) that just comparing raw page text was
        not reliable enough: Workday's own page re-renders something trivial
        (tooltip/focus state) within 1-2 seconds on its own, with nobody
        touching the page, which looked like "the page changed" and made the
        bot resume immediately and re-attempt the exact same broken fields —
        over and over, every few seconds, never actually stopping to wait
        for a real human action. Checking that the specific validation
        errors are gone (not just that "some text" changed) is a much
        stronger signal that you genuinely completed the step.
        """
        try:
            handles_before_wait = self.driver.window_handles
        except Exception:
            handles_before_wait = []

        self._show_stuck_banner(message)
        while True:
            # Lets a Skip click take effect immediately even while stuck
            # here — previously nothing checked for it until fill_form()
            # returned and REVIEW REQUIRED was reached, so a job stuck in
            # this exact loop had no way to be skipped at all. Caught and
            # handled in apply_to_job(). Mirrors the same addition in
            # dice_apply.py.
            try:
                from src.workflow.apply import skip_event, SkipRequested, check_browser_alive, BrowserClosedByUser
                if skip_event.is_set():
                    raise SkipRequested()
                check_browser_alive(self.driver)
            except ImportError:
                pass

            # A brand-new tab/window can open for the real form WITHOUT the
            # interim tab ever closing (same pattern already handled at the
            # top of fill_form()) — checked first, before either of the
            # dead-window checks below, so it's never mistaken for a closed
            # browser. Mirrors builtin_apply.py, which already had this.
            try:
                current_handles = self.driver.window_handles
            except Exception:
                current_handles = []
            new_handles = [h for h in current_handles if h not in handles_before_wait]
            if new_handles:
                try:
                    self.driver.switch_to.window(new_handles[-1])
                    self._log("A new tab/window opened — switched to it to resume filling.")
                    break
                except Exception:
                    pass

            try:
                self.driver.current_window_handle
                window_alive = True
            except Exception:
                window_alive = False

            if not window_alive:
                if self._recover_dead_window():
                    break
                if not current_handles:
                    # Not a recoverable single-tab closure (_recover_dead_
                    # window() already tried switching to a surviving tab and
                    # found none) — every window is genuinely gone. Confirmed
                    # live this was a real, serious gap: without this check,
                    # _page_signature()/_detect_validation_errors() below both
                    # fail safe to ""/[] on a dead browser, which looks
                    # IDENTICAL to "the page genuinely changed and has no
                    # errors" — so the loop wrongly concluded you'd finished
                    # the step and resumed filling against a browser that no
                    # longer existed, eventually reaching a false "form fully
                    # filled" state that got marked Submitted despite nothing
                    # real ever being filled or submitted. Raising here lets
                    # it propagate as an honest "not submitted" stop instead.
                    raise BrowserClosedByUser("Browser window closed while waiting for manual step completion")
                time.sleep(2)
                continue

            try:
                current_signature = self._page_signature()
            except Exception:
                current_signature = page_signature_before
            if current_signature != page_signature_before and not self._detect_validation_errors():
                break
            time.sleep(2)
        self._hide_stuck_banner()
        # A page-signature change on a slow-hydrating ATS (Workday confirmed
        # live) can just mean a loading state replaced the previous text —
        # not that the real form has actually rendered yet. Waiting here for
        # genuine interactive elements (same bar used everywhere else in this
        # file) before declaring "resumed" avoids immediately running a fill
        # pass against a still-empty page and concluding there's nothing left
        # to do.
        self.wait_for_page_ready(timeout=12)
        self._just_resumed_from_human = True
        self._log("Page changed and no validation errors remain — resuming automatic filling.")

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
            job_dir = os.path.join(APPLICATIONS_DIR, self.job_id or "unknown-job")
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
                # Real per-character keystrokes first, same as every other
                # LLM-answered text field in _apply_llm_field_answers() — confirmed
                # live on a FOX Careers / Workday signup form: the JS native-setter
                # path used here previously left both password fields showing the
                # correct value in the DOM, yet the "Create Account" button's own
                # click still did nothing at all, even after waiting — consistent
                # with a React/Formik-style form whose internal validity state
                # didn't get updated because no real input events were ever fired.
                # Falls back to the JS setter only if send_keys didn't actually
                # take (e.g. a masked/custom widget intercepting real keystrokes).
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
            self._log(f"Generated a strong signup password and saved it to {saved_path}", level="WARN")
        else:
            self._log("Generated a strong signup password but could not save it to a file — "
                       "check the console/log output for it: " + password, level="WARN")

        filled_any = self._fill_all_password_fields(password)
        if filled_any:
            self._log("Filled the generated password into the signup form.")
        return filled_any

    def _read_saved_password(self):
        """
        Reads back the most recently generated password for THIS job from
        its saved credentials file — used when the bot lands on a Sign-In
        gate (not a fresh signup) so it doesn't generate and fill a brand
        new, wrong password that could never match the account that was
        actually created earlier in this same run.
        """
        try:
            path = os.path.join(APPLICATIONS_DIR, self.job_id or "unknown-job", "account_credentials.txt")
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
        a genuine sign-in gate apart from a normal signup/continue step."""
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

    # Some ATS portals insert a bare Terms-of-Use / consent interstitial between
    # the landing-page click and the real application form — confirmed live on
    # FOX Careers (fronts a Workday application): a page with NO form fields at
    # all, just legal boilerplate text and a single
    # '<input type="submit" id="AgreementButton" value="Submit">' button. The
    # old code treated that lone "Submit" button as _is_final_submit_step() ==
    # True and returned "review_reached" immediately — permanently stopping
    # fill_form() before the real Workday form was ever reached, which is why
    # the form appeared to "do nothing" even after a human clicked through the
    # gate manually (the function had already returned; nothing was running
    # anymore). Deliberately narrow: only fires when the page has literally NO
    # form-field content (_frame_has_form_content() False) AND the visible text
    # contains one of these unambiguous consent/ToU phrases — a genuine
    # "ready to submit" review page always either still shows its filled-in
    # fields or won't match these phrases, so this can never be confused with
    # an actual application submission.
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

    def _click_bare_agreement_gate(self):
        """If _is_bare_agreement_gate() is True, clicks the lone Agree/Accept/
        Acknowledge button so the flow can proceed to the real form. Returns
        False (no-op) on any page that has real form content, or that doesn't
        match the consent-page text markers — never touches an actual submit
        step.

        "submit" is deliberately excluded from the clickable-text list below
        (even though the original FOX Careers ToU gate this was built for
        used a literal "Submit"-labeled button) — confirmed live on two real
        Workday jobs: a genuine final review/submit page can also have
        _frame_has_form_content() == False (previously-entered data shown as
        read-only summary text, not live fields) AND boilerplate text
        mentioning "Privacy Policy"/"Terms of Use" near the Submit button
        (an almost universal pattern on final application pages), which
        together satisfy _is_bare_agreement_gate()'s trigger — so this was
        auto-clicking the REAL application Submit button, not just a consent
        gate. A literal "Submit"-labeled ToU-only gate (no real application
        content at all) now correctly falls through to _is_final_submit_step()
        instead and pauses for you — one extra manual step in that narrow
        case, instead of ever auto-submitting a real application.
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
                        self._log(f"Clicked through an intermediate Terms-of-Use/agreement gate: '{text}'")
                        return True
            except Exception:
                pass
        return False

    def _is_final_submit_step(self):
        try:
            selectors = [
                "xpath://button",
                "input[type='submit']",
                # Same Workday div/span[role='button'] pattern _CONTINUE_BUTTON_SELECTORS
                # had to add — a real final Submit rendered that way must also be
                # recognized here, or a genuinely-finished page falls through to the
                # generic "no button found" path instead of this one.
                "div[role='button']",
                "span[role='button']",
            ]
            for sel in selectors:
                by_type = By.XPATH if sel.startswith("xpath:") else By.CSS_SELECTOR
                actual_sel = sel[6:] if sel.startswith("xpath:") else sel
                for el in self.driver.find_elements(by_type, actual_sel):
                    aria_disabled = (el.get_attribute("aria-disabled") or "").lower() == "true"
                    if el.is_displayed() and el.is_enabled() and not aria_disabled:
                        text = (el.text or el.get_attribute("value") or "").strip().lower()
                        if "next" in text or "continue" in text:
                            continue
                        if any(k in text for k in self._FINAL_STEP_KEYWORDS):
                            return True
        except Exception:
            pass
        return False

    # Some ATS forms overlay their real, interactive button with a second
    # decoy/tracking element positioned exactly on top of it — confirmed live
    # on a FOX Careers / Workday "Create Account" step: the real
    # <button data-automation-id="createAccountSubmitButton"> is
    # aria-hidden="true" and tabindex="-2" (i.e. genuinely non-interactive),
    # while a <div role="button" data-automation-id="click_filter"
    # aria-label="Create Account"> sits visually on top of it and is what
    # Workday's own click handler is actually wired to. Selenium's real
    # el.click() correctly raised ElementClickIntercepted here — but the old
    # exception fallback (a blind execute_script("arguments[0].click()"))
    # bypassed that check entirely and force-clicked the hidden, non-
    # functional button underneath, which is why the page never advanced
    # despite every field being filled correctly and the click "succeeding".
    # This clicks whatever element is ACTUALLY at that point on screen
    # instead — the same thing a real mouse click would hit.
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

    _CONTINUE_BUTTON_SELECTORS = [
        "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue')]",
        "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'next')]",
        "xpath://a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue')]",
        "button[type='submit']",
        # Workday's Canvas Kit UI (confirmed live on a PwC "Managed Services -
        # Data Engineer" job reached via LinkedIn's external-apply path) renders
        # its footer "Save and Continue" as a
        # <div role="button" data-automation-id="bottom-navigation-next-button">,
        # never a real <button>/<a> — the selectors above matched nothing on that
        # page, so both this and _click_continue_button() fell all the way through
        # to "no button found, must be finished", silently stopping one full page
        # early with fields already correctly filled but never submitted forward.
        # Text-gated exactly like the plain-<button> xpaths above (never a bare
        # `div[role='button']`, which would match unrelated on-page controls too).
        "xpath://div[@role='button'][contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'next')]",
        "xpath://span[@role='button'][contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'next')]",
        "[data-automation-id*='next' i]",
        "[data-automation-id*='continue' i]",
    ]

    def _has_disabled_continue_button(self):
        """
        True only if a Continue/Next/Save-and-Continue-style button actually
        EXISTS and is visible but currently disabled — a specific signal that
        a required field is blocking progress (confirmed live on Elevance
        Health / Workday: "Save and Continue" stayed disabled because a
        custom dropdown and Yes/No question weren't recognized as fillable).

        Deliberately narrower than "no button was clickable" — a page that
        has genuinely finished and simply has no such button at all (no
        matches here, not a disabled one) must NOT trigger this, or it
        reintroduces the exact indefinite-hang bug fixed earlier: a
        synthetic/real form with zero submit controls has nothing to ever
        wait for, so escalating to wait-for-human on it would hang forever.
        """
        for sel in self._CONTINUE_BUTTON_SELECTORS:
            try:
                by_type = By.XPATH if sel.startswith("xpath:") else By.CSS_SELECTOR
                actual_sel = sel[6:] if sel.startswith("xpath:") else sel
                for el in self.driver.find_elements(by_type, actual_sel):
                    # is_enabled() only reflects the DOM `disabled` property, which
                    # non-form elements (the div/span[role='button'] cases above)
                    # never have — Workday marks those disabled via aria-disabled
                    # instead, so that must be checked too or a genuinely-blocked
                    # custom button here is invisible to this check.
                    aria_disabled = (el.get_attribute("aria-disabled") or "").lower() == "true"
                    if el.is_displayed() and (not el.is_enabled() or aria_disabled):
                        return True
            except Exception:
                pass
        return False

    def _has_required_empty_file_field(self):
        """
        True if a visible, required file-upload field (Resume/CV, etc.) is
        still empty — a signal that the step is genuinely NOT finished yet,
        even when no Continue/Submit button exists at all. Confirmed live on
        a New York Power Authority job: the page had no button matching any
        Continue-button selector (its buttons are labeled "Apply"/"Save"
        instead), so the old logic fell straight through to "no further
        button found, treating as final" and silently abandoned the job with
        a required Resume field still empty. Deliberately narrow — only
        fires when a field is both unfilled (no data-bot-uploaded marker,
        i.e. _try_upload_file() never succeeded on it) AND flagged required
        (native `required` attribute, or a nearby "*" marker next to
        resume/cv/cover text, matching the red-asterisk convention many ATS
        forms use) — never on a form that's genuinely finished.
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
        for sel in self._CONTINUE_BUTTON_SELECTORS:
            try:
                by_type = By.XPATH if sel.startswith("xpath:") else By.CSS_SELECTOR
                actual_sel = sel[6:] if sel.startswith("xpath:") else sel
                for el in self.driver.find_elements(by_type, actual_sel):
                    aria_disabled = (el.get_attribute("aria-disabled") or "").lower() == "true"
                    if not (el.is_displayed() and el.is_enabled()) or aria_disabled:
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
                    if "submit" in text:
                        continue
                    if "continue" not in text and "next" not in text and \
                            any(k in text for k in self._FINAL_STEP_KEYWORDS):
                        continue
                    # Confirmed live on a GM Motorsports / Workday job: an
                    # "Already have an account? Sign In" link/button next to
                    # the real Create Account button has type='submit' too,
                    # so it was being caught by the button[type='submit']
                    # catch-all above and clicked instead of the real
                    # continue action — navigating to a login page the bot
                    # has no credentials for, instead of completing account
                    # creation. Mirrors the same fix already in
                    # dice_apply.py (found live on Randstad/HTC jobs). An
                    # auth-gate button is never a legitimate "advance the
                    # form" click.
                    if any(k in text for k in ("log in", "login", "sign in", "sign-in")):
                        continue
                    self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                    time.sleep(0.2)
                    self._click_resilient(el)
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
        self._fill_known_custom_dropdowns()
        self._fill_date_inputs()
        self._fill_today_date_fields()
        self._fill_range_sliders()
        self._fill_skills_field(cv_data)

        page_text = self._page_signature()
        remaining_fields = self._collect_unfilled_fields()
        if remaining_fields:
            answers = self._llm_answer_fields(cv_data, remaining_fields, page_text)
            if answers:
                self._apply_llm_field_answers(answers)
                self._log(f"LLM answered {len(answers)} additional field(s) (of {len(remaining_fields)} detected)")
        return identity_filled, len(remaining_fields)

    # ------------------------------------------------------------------ main loop

    def fill_form(self, cv_data, cover_letter, max_steps=20):
        # Don't fill until the page has actually settled.
        self.wait_for_page_ready(timeout=12)
        # Checked before the iframe search below — a CAPTCHA/bot-check challenge
        # (e.g. DataDome) can occupy the page where the real form would otherwise
        # be, which _switch_into_form_iframe_if_needed() has no way to distinguish
        # from "this site just uses an unrecognized form layout" on its own.
        # No-op if no such challenge is present.
        self.wait_for_bot_challenge()

        # Some ATS forms (Greenhouse's embeddable widget confirmed live) are inside
        # an <iframe> on the company's own page rather than served directly — must
        # happen BEFORE the landing-page-click check below, since the real
        # Apply/Continue button (if any) usually lives inside that iframe too, not
        # on the outer page.
        self._switch_into_form_iframe_if_needed()
        # Some redirects land on an intermediate job-repost/landing page (not the
        # form itself) that needs its own Apply/Continue click first.
        self._landing_apply_click_happened = False
        self._click_landing_page_apply_if_needed()
        real_click_happened = self._landing_apply_click_happened

        # Only verify/escalate when a click actually happened (something
        # matched "apply" and got clicked) but didn't lead anywhere real —
        # NOT when nothing was found to click at all. A page with genuinely
        # no apply button and no form (confirmed live on a MANTECH job used
        # in regression testing) must keep falling through to the existing
        # "no button found" handling unchanged; escalating on that case too
        # would wait forever for a human to click a button that was never
        # there to begin with. Deliberately keyed off whether a REAL click
        # happened (not off whether the page already looked like it had a
        # form before the click) — confirmed live on SIERTEK LTD: form-like
        # fields were already visible (an unrelated quick-apply/search
        # widget), yet the bot still went on to click a distinct real Apply
        # button, and that click needs the exact same verification as the
        # "no form at all yet" case, not to be silently skipped.
        if real_click_happened:
            # Verify that click actually led to a real application form before
            # trusting it and filling anything — confirmed live on Yamaha Motor
            # Corporation: the generic "apply"-text matcher grabbed the site's
            # own top-nav "APPLY NOW" menu item (a persistent header link, not
            # a job-specific action) instead of the real per-job "Apply" link
            # further down the page, revealing the site's own job-SEARCH bar
            # (Keyword + Location fields) — which the bot then filled in as if
            # it were the real application and declared itself done, without
            # ever actually applying. Rather than trying to out-guess every
            # possible site's DOM structure, this tries automatically exactly
            # once and, if that didn't lead to a genuine form, asks you to
            # find and click the real Apply button yourself — the bot then
            # resumes automatically once a real form appears, same as every
            # other wait-for-human step.
            landed_on_real_form = False
            try:
                landed_on_real_form = self._frame_has_form_content()
            except Exception:
                pass
            if not landed_on_real_form:
                self._log("This step needs your help — couldn't confirm a real application form "
                           "opened. Please find and click the Apply button yourself; "
                           "the bot will resume automatically once the form appears.", level="WARN")
                self._wait_for_human_to_advance_step(
                    self._page_signature(),
                    message="Bot failed — please try manually.",
                )

        self._just_resumed_from_human = False
        for step in range(max_steps):
            try:
                from src.workflow.apply import skip_event, SkipRequested, check_browser_alive
                if skip_event.is_set():
                    raise SkipRequested()
                check_browser_alive(self.driver)
            except ImportError:
                pass
            identity_filled, remaining_count = self._run_all_fill_passes(cv_data, cover_letter)

            # Also applies right after a human-assisted resume, not just on the
            # form's very first step — confirmed live on an Ovative Group /
            # Workday job: resuming after you complete a step manually can land
            # back here while the next (heavy, slow-hydrating) Workday screen is
            # still rendering, and without this the bot would give up on it
            # immediately instead of getting the same one-retry cushion the
            # first step already gets.
            just_resumed = self._just_resumed_from_human
            self._just_resumed_from_human = False
            if (step == 0 or just_resumed) and identity_filled == 0 and remaining_count == 0:
                self._log("No identity fields and no other unfilled fields matched on first pass — "
                           "waiting and retrying once in case the page was still rendering...", level="WARN")
                time.sleep(2.5)
                identity_filled, remaining_count = self._run_all_fill_passes(cv_data, cover_letter)

            # Checked before the final-step/continue-click logic below — an empty
            # password field means the form is genuinely not done yet, regardless
            # of what the Continue button's state looks like (confirmed live on
            # hackajob: the button was simply unclickable until this was filled,
            # which the old code misread as "nothing left to do"). Auto-fills a
            # generated password instead of waiting on you to type one; only
            # falls back to waiting if the auto-fill didn't actually take.
            if self._has_unfilled_password_field():
                if self._has_auth_gate_button():
                    # A Sign-In gate, not a fresh signup — confirmed live on a
                    # GM Motorsports / Workday job that already had an
                    # account from an earlier test run of this same job:
                    # generating a brand-new random password here would be
                    # wrong (it could never match the account that was
                    # actually created), so re-use the one already generated
                    # and saved for this job instead. The email is
                    # deliberately left for you to fill — the bot has no
                    # reliable way to know whether this is really the same
                    # account/email the credentials were saved under — and
                    # the actual Sign In click is always a human action (see
                    # the login-gate exclusion in _click_continue_button).
                    saved_password = self._read_saved_password()
                    if saved_password and self._fill_all_password_fields(saved_password):
                        self._log("Reached a sign-in page — filled in your previously generated password. "
                                   "Please enter your email and click Sign In; "
                                   "the bot will resume automatically afterward.", level="WARN")
                    else:
                        self._log("Reached a sign-in page but no saved password was found for this job — "
                                   "please enter your email and password and sign in; "
                                   "the bot will resume automatically afterward.", level="WARN")
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
                self._log("Form fully filled — reached final submit step. Waiting for your review/submission.")
                return "review_reached"

            text_before = self._page_signature()
            clicked = self._click_continue_button()
            if not clicked:
                # No Continue/Next button here — but that can also mean this isn't
                # a form step at all, just another landing/launcher screen further
                # into the flow (confirmed live on FOX Careers: after its
                # Terms-of-Use gate, the real Workday job page shows an
                # "Apply Manually" link — not a Continue button — which is what
                # actually starts the real application; the very same
                # landing-page-apply pattern already handled once at the top of
                # fill_form(), just needing another attempt now that we're deeper
                # into the flow). Safe to retry here specifically because we've
                # already ruled out both a real Continue/Next button (just above)
                # and a real final-submit button (_is_final_submit_step(), above)
                # — a genuine form step always has one or the other.
                # _click_landing_page_apply_if_needed() is a pure no-op
                # (nothing to search for or click) whenever real form
                # content is already visible — confirmed via a regression
                # test with a synthetic form that has no Continue/Submit
                # button at all: counting that no-op as a "retry attempt"
                # below caused the bot to eventually escalate to
                # _wait_for_human_to_advance_step(), which then waited
                # forever for a page that has genuinely nothing left to
                # click, ever. Only worth retrying — and eventually pausing
                # for help on — when an ACTUAL search-and-click attempt is
                # happening.
                already_has_form = False
                try:
                    already_has_form = self._frame_has_form_content()
                except Exception:
                    pass

                # Specifically "a Continue/Save button exists and is visible
                # but disabled" — NOT the broader "form visible, nothing
                # clicked" — confirmed live on Elevance Health / Workday: a
                # custom dropdown and Yes/No question weren't recognized as
                # fillable, so "Save and Continue" stayed disabled and the
                # page was silently declared "final step" without ever
                # pausing for help. Must stay this narrow: a form that is
                # genuinely finished and simply has no submit control at all
                # (no matches here at all, not a disabled one) would hang
                # forever if this escalated on that case too — the same
                # indefinite-hang bug already fixed once before.
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

                if has_disabled_continue or has_required_empty_file:
                    if has_required_empty_file:
                        self._log("This step needs your help — a required file field (e.g. Resume/CV) "
                                   "is still empty and no Continue button was found. Please upload it "
                                   "yourself and click Next/Continue/Submit; the bot will resume "
                                   "automatically once the page advances.", level="WARN")
                    else:
                        self._log("This step needs your help — a Continue/Submit button is disabled, "
                                   "likely blocked by an unrecognized or unfilled required field. "
                                   "Please complete it and click Next/Continue/Submit yourself; "
                                   "the bot will resume automatically once the page advances.", level="WARN")
                    self._wait_for_human_to_advance_step(
                        self._page_signature(),
                        message="Bot failed — please try manually.",
                    )
                    continue

                if not already_has_form and self._click_landing_page_apply_if_needed():
                    time.sleep(1.2)
                    # Only one automatic attempt — confirmed live on Yamaha
                    # Motor Corporation and a GM Motorsports / Workday job:
                    # letting this retry itself a couple more times either
                    # kept re-clicking the same non-advancing element loop
                    # after loop, silently, or (Yamaha) grabbed the wrong
                    # "apply"-labeled element (a site's own nav menu item)
                    # and treated whatever it revealed as a real form. Safer
                    # to hand off to you immediately after a single try than
                    # to keep guessing — the bot resumes automatically once
                    # you get it to a real form.
                    self._landing_retry_count = getattr(self, "_landing_retry_count", 0) + 1
                    if self._landing_retry_count <= 0:
                        continue
                    self._log("This step needs your help — repeated automatic clicks aren't moving the form "
                               "forward. Please complete it and click Next/Continue/Submit yourself; "
                               "the bot will resume automatically once the page advances.", level="WARN")
                    self._wait_for_human_to_advance_step(
                        self._page_signature(),
                        message="Bot failed — please try manually.",
                    )
                    self._landing_retry_count = 0
                    continue

                # A plain "Login"/"Sign in" link is present on almost every corporate
                # careers page as normal top-nav ("already have an account?") chrome, so
                # only treat it as a genuine blocking gate (the Randstad/HTC Global case
                # this was built for, where the login prompt was the only thing on the
                # page) when there's also no other real fillable content left —
                # otherwise there's clearly more to do here first. Mirrors
                # builtin_apply.py, which already had this check; this file was
                # previously missing it and would silently fall through to the
                # generic "treating current step as final" message below instead.
                auth_gate_visible = False
                try:
                    auth_gate_visible = self._has_auth_gate_button()
                except Exception:
                    pass
                still_unfilled = []
                if auth_gate_visible:
                    try:
                        still_unfilled = self._collect_unfilled_fields()
                    except Exception:
                        pass
                if auth_gate_visible and not still_unfilled:
                    # Confirmed live on Randstad Digital / HTC Global Services jobs
                    # (Ceipal-style ATS): the portal gates the real application behind
                    # a Login/Sign-in step that only the candidate's own credentials or
                    # email verification can get past — the bot correctly refuses to
                    # click it (see _click_continue_button), but must say so plainly
                    # instead of implying the form is actually complete when only 1-2
                    # fields got filled.
                    self._log("This application requires signing in / verifying your email on the "
                               "employer's portal — only a login gate is visible, not the real "
                               "application form. Please complete this step manually.", level="WARN")
                    return "review_reached"

                self._log("No further Continue/Next button found — treating current step as final.")
                return "review_reached"

            self._landing_retry_count = 0

            time.sleep(1.8)
            self.wait_for_cloudflare()
            self.wait_for_page_ready(timeout=8)

            # One clean fill pass, one click, per step — never re-filling/
            # re-clicking the same step in a blind loop. If the click didn't
            # genuinely move the form forward — either nothing changed at all,
            # or it changed only to show validation errors (confirmed live on
            # a Workday-based application: clicking "Save and Continue" with a
            # couple of unrecognized custom fields still blank redraws the
            # SAME page with red error banners, which IS a page-signature
            # change, so the old check here missed it and kept re-attempting
            # the same broken fields up to max_steps) — stop trying to fix it
            # automatically and wait for you to finish this one step and
            # click Next/Continue/Submit yourself. The bot then resumes
            # automatically for every step after it, all the way through to
            # the real final submit step.
            page_after_click = self._page_signature()
            validation_errors = self._detect_validation_errors()
            if page_after_click == text_before or validation_errors:
                detail = (" Validation errors found: " + " | ".join(validation_errors[:5])) if validation_errors else \
                          " No validation-error indicators found — likely an unrecognized custom field is still blocking progress."
                self._log("This step needs your help —" + detail +
                           " Please complete it and click Next/Continue/Submit yourself; "
                           "the bot will resume automatically once the page advances.", level="WARN")
                self._wait_for_human_to_advance_step(
                    page_after_click,
                    message="Bot failed — please try manually.",
                )
                continue

        self._log("Reached max auto-fill steps — leaving form for manual completion/review.", level="WARN")
        return "max_steps"

    def run(self, job_id: str):
        self.job_id = job_id
        self._log("Handling external redirect...")

        time.sleep(3.5)
        new_handles = self.driver.window_handles
        if len(new_handles) > len(self.original_handles):
            new_tab = [h for h in new_handles if h not in self.original_handles][-1]
            self.driver.switch_to.window(new_tab)
            self._log("Switched to new tab/window for external LinkedIn application.")
        else:
            self._log("No new tab detected, continuing in current tab.")

        # Safety net for the same tab-closes-itself pattern _recover_dead_window()
        # handles below — cheap no-op if the current tab is still fine.
        self._recover_dead_window()

        self.wait_for_cloudflare()
        return self.fill_form(self.cv_data, self.cover_letter)
