"""
Standalone, isolated testbed for Dice EXTERNAL apply-form handling — mirrors
linkedin_test_apply.py's approach, but for jobs sourced from Dice instead of
LinkedIn. Does not import from, or modify, dice_apply.py in any way; all new
work for Dice testing lives in this file only.

Reuses the SAME generic form-filling engine (LinkedInExternalTestApplier) that
was already built and verified against Ladders, a Dice/Ceipal job, and a
Fastly/Greenhouse (iframe-embedded) job — the only genuinely source-specific
part is how you get from a Dice job listing to the actual application form,
which is what DiceTestApplier adds. Sharing one engine instead of copy-pasting
it again means a future fix to the field-detection/answering logic benefits
both LinkedIn and Dice testing at once, instead of needing to be applied twice
and slowly drifting out of sync.

Usage:
    python -m src.workflow.dice_test_apply "<dice job url>"
"""
import os
import sys
import time

from selenium.webdriver.common.by import By

from src.workflow.linkedin_test_apply import (
    LinkedInExternalTestApplier,
    _latest_uploaded_resume,
    _load_cv_and_cover_letter,
    _init_test_driver,
)


class DiceTestApplier(LinkedInExternalTestApplier):
    # Fixes a false-positive confirmed live on a real Randstad-hosted Dice job:
    # the job LISTING page (before the real application is ever reached) has its
    # own "get job alerts" widget — search keyword, location, email, frequency
    # dropdown — which is 4 form-shaped fields that satisfy the inherited
    # "does this page already have a real form" check, so the code concluded a
    # form already existed here and never clicked the real "apply" link sitting
    # right next to it. General fix (not Randstad-specific): a real application
    # almost always has a resume upload or a clearly-named first/last-name/phone
    # field — a job-alert/search widget never does — so that's checked first as
    # a strong, specific signal; the same known non-application keywords (search,
    # alert, subscribe, frequency, etc.) are also stripped out of the weaker
    # generic-field-count fallback so an unrelated site widget can't pad the count.
    _NON_APPLICATION_FIELD_KEYWORDS = (
        "search", "alert", "subscribe", "newsletter", "query", "frequency", "vendor", "keyword", "signup",
    )
    _APPLICATION_IDENTITY_SELECTOR = (
        "input[type='file'], "
        "input[name*='first' i], input[id*='first' i], "
        "input[name*='last' i], input[id*='last' i], "
        "input[name*='resume' i], input[id*='resume' i], "
        "input[name*='phone' i], input[id*='phone' i], input[type='tel']"
    )

    def _frame_has_form_content(self):
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

    def _click_landing_page_apply_if_needed(self):
        """Same logic as the inherited version, but checking through the
        overridden _frame_has_form_content() above instead of the parent's own
        inline probe (which has the same false-positive the override fixes)."""
        if self._frame_has_form_content():
            return True

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

    _DICE_APPLY_SELECTORS = [
        "xpath://a[contains(., 'Easy Apply')]", "xpath://button[contains(., 'Easy Apply')]",
        "xpath://a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'apply')]",
        "xpath://button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'apply')]",
    ]

    def open_and_click_apply_dice(self, job_url):
        """
        Dice-specific entry point — everything after this (fill_form) is the
        same shared engine used for LinkedIn external forms. Dice has two
        distinct apply flows from the same "Apply" click depending on the job:
          - Dice's OWN hosted wizard (dice.com/job-applications/<id>/wizard,
            same tab) — already handled well by the working, protected
            DiceApplier in dice_apply.py. Out of scope here; this bails out
            cleanly rather than duplicating or interfering with that flow.
          - An external redirect to the company's own site (opens a new tab)
            — this is what actually needs the generic engine, same as
            LinkedIn's external-apply jobs.
        """
        self.driver.get(job_url)
        time.sleep(3)

        handles_before = self.driver.window_handles
        clicked = self._click(self._DICE_APPLY_SELECTORS, "Dice Apply button", timeout=10)
        if not clicked:
            self._log("Could not find an Apply/Easy Apply button on this Dice job page.", level="WARN")
            return False

        time.sleep(2.5)
        new_handles = self.driver.window_handles
        if len(new_handles) > len(handles_before):
            new_tab = [h for h in new_handles if h not in handles_before][-1]
            self.driver.switch_to.window(new_tab)
            self._log("Switched to new tab/window for external application.")
            self.wait_for_cloudflare()
            return True

        # No new tab — check whether this landed on Dice's own native wizard
        # rather than a same-tab external redirect (both are technically
        # possible without a new tab, but the wizard URL is a reliable tell).
        self.wait_for_cloudflare()
        if "dice.com" in self.driver.current_url.lower() and "/wizard" in self.driver.current_url.lower():
            self._log("This job uses Dice's own native Easy Apply wizard, not an external form — "
                       "this tool only targets external apply-form handling (Dice's own wizard is "
                       "already handled by the existing, working DiceApplier). Pick a job where Dice "
                       "redirects to an external company site instead.", level="WARN")
            return False

        self._log("No new tab detected — continuing in current tab (same-tab redirect).")
        return True

    def run(self, job_url):
        if not self.open_and_click_apply_dice(job_url):
            return "could_not_start"
        return self.fill_form(self.cv_data, self.cover_letter)


def run_test(job_url, job_description=""):
    cv_data, cover_letter = _load_cv_and_cover_letter(job_description)
    cv_filepath = _latest_uploaded_resume()
    if not cv_filepath:
        print("[WARN] No uploaded resume file found in data/uploads — resume upload will be skipped.")

    driver = _init_test_driver()
    applier = DiceTestApplier(driver, cv_data, cover_letter, cv_filepath, job_id="dice-test")
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
        print("Usage: python -m src.workflow.dice_test_apply \"<dice job url>\" [job_description.txt]")
        sys.exit(1)

    url = sys.argv[1]
    description = ""
    if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            description = f.read()

    run_test(url, description)
