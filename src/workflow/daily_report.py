"""
Daily job-report emailer.

Invoked once per day by a Windows Scheduled Task (fixed local trigger time
equivalent to 12:00 AM Pakistan Standard Time). PKT is a constant UTC+5
offset with no daylight-saving shifts, so it's hardcoded here via the
stdlib `timezone` class rather than a full IANA tz database.

Manual runs:
    python src/workflow/daily_report.py                      # send today's scheduled report (yesterday's data, PKT)
    python src/workflow/daily_report.py --force               # resend even if already sent today
    python src/workflow/daily_report.py --force --date 2026-07-02
"""
import os
import sys
import json
import glob
import smtplib
import argparse
from datetime import datetime, timedelta, timezone, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import urlsplit, urlunsplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src import config
from src.storage.tracker import DATA_DIR, TRACKER_FILE, JobStatus
from src.utils.logger import log, error

PKT = timezone(timedelta(hours=5))  # Pakistan Standard Time — fixed offset, no DST
LAST_SENT_FILE = os.path.join(DATA_DIR, "last_report_sent.json")

STATUS_LABELS = {
    JobStatus.SUBMITTED: "Applied",
    JobStatus.SKIPPED: "Skipped",
    JobStatus.EXPIRED: "No Longer Accepting Applications",
    JobStatus.DISCOVERED: "Not Applied",
    # A job left in TAILORED status means an apply attempt was made (browser opened,
    # form filled) but never reached submission — either the user closed the browser
    # or the 180s review window timed out. Functionally that's a skip by day's end.
    JobStatus.TAILORED: "Skipped",
    JobStatus.READY: "Not Applied",
    JobStatus.FAILED: "Not Applied",
}
NOT_APPLIED_LABEL = "Not Applied"


def _canonical_job_url(url):
    if not url:
        return ""
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _file_date_pkt(filepath):
    """PKT calendar date embedded in a jobs-*/summary-* filename (UTC timestamp)."""
    base = os.path.basename(filepath)
    stamp = base.rsplit("-", 1)[-1].replace(".json", "")
    try:
        dt_utc = datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        return dt_utc.astimezone(PKT).date()
    except ValueError:
        return None


def _load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _collect_scraped_jobs(target_date: date):
    """Unique jobs (deduped by canonical URL) scraped on the given PKT date."""
    files = glob.glob(os.path.join(DATA_DIR, "jobs-unique-*.json"))
    matching = [f for f in files if _file_date_pkt(f) == target_date]

    seen = set()
    jobs = []
    for filepath in matching:
        for job in _load_json(filepath, []):
            key = _canonical_job_url(job.get("url")) or f"{job.get('source', '')}::{job.get('job_title', '')}::{job.get('company', '')}"
            if key in seen:
                continue
            seen.add(key)
            jobs.append(job)
    return jobs


def _raw_discovered_count(target_date: date):
    """Total raw (pre-dedup) discovered count for the day, from summary files."""
    files = glob.glob(os.path.join(DATA_DIR, "summary-*.json"))
    total = 0
    for filepath in files:
        if _file_date_pkt(filepath) != target_date:
            continue
        summary = _load_json(filepath, {})
        total += summary.get("total_discovered", 0)
    return total


def _tracker_status_map():
    tracker = _load_json(TRACKER_FILE, {"applications": []})
    status_map = {}
    for entry in tracker.get("applications", []):
        jid = str(entry.get("job_id") or "")
        if jid:
            status_map[jid] = entry
    return status_map


def build_report(target_date: date):
    scraped_jobs = _collect_scraped_jobs(target_date)
    status_map = _tracker_status_map()

    rows = []
    counts = {"Applied": 0, "Skipped": 0, "No Longer Accepting Applications": 0, "Not Applied": 0}

    for job in scraped_jobs:
        jid = str(job.get("id") or job.get("job_id") or "")
        entry = status_map.get(jid)
        raw_status = entry.get("status") if entry else None
        label = STATUS_LABELS.get(raw_status, NOT_APPLIED_LABEL)
        counts[label] += 1
        rows.append({
            "job_title": job.get("job_title", "Unknown"),
            "company": job.get("company", "Unknown"),
            "source": job.get("source", ""),
            "status": label,
            "updated_at": entry.get("updated_at") if entry else "",
        })

    rows.sort(key=lambda r: (r["status"] != "Applied", r["status"] != "No Longer Accepting Applications", r["job_title"].lower()))

    summary = {
        "date": target_date.isoformat(),
        "total_raw_discovered": _raw_discovered_count(target_date),
        "total_scraped": len(scraped_jobs),
        "applied": counts["Applied"],
        "skipped": counts["Skipped"],
        "expired": counts["No Longer Accepting Applications"],
        "not_applied": counts["Not Applied"],
    }
    return summary, rows


def _format_pkt(iso_string):
    if not iso_string:
        return ""
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        return dt.astimezone(PKT).strftime("%Y-%m-%d %I:%M %p PKT")
    except ValueError:
        return iso_string


def render_html(summary, rows):
    status_colors = {
        "Applied": "#10b981",
        "Skipped": "#f59e0b",
        "No Longer Accepting Applications": "#ef4444",
        "Not Applied": "#64748b",
    }

    def row_html(r):
        color = status_colors.get(r["status"], "#64748b")
        return f"""
        <tr>
          <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;">{r['job_title']}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;">{r['company']}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;">{r['source']}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;color:{color};font-weight:600;">{r['status']}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;">{_format_pkt(r['updated_at'])}</td>
        </tr>"""

    rows_html = "".join(row_html(r) for r in rows) or (
        '<tr><td colspan="5" style="padding:16px;text-align:center;color:#64748b;">'
        'No jobs were scraped on this date.</td></tr>'
    )

    return f"""
    <html>
    <body style="font-family:Segoe UI,Arial,sans-serif;background:#f8fafc;padding:24px;color:#0f172a;">
      <div style="max-width:760px;margin:0 auto;background:#ffffff;border-radius:12px;overflow:hidden;border:1px solid #e2e8f0;">
        <div style="background:#0f172a;padding:20px 24px;">
          <h2 style="color:#ffffff;margin:0;">Job Bot &mdash; Daily Report</h2>
          <p style="color:#94a3b8;margin:4px 0 0;">{summary['date']} (Pakistan Standard Time)</p>
        </div>
        <div style="padding:24px;">
          <table style="width:100%;border-collapse:collapse;margin-bottom:24px;">
            <tr>
              <td style="padding:10px;text-align:center;background:#f1f5f9;border-radius:8px 0 0 8px;">
                <div style="font-size:22px;font-weight:700;">{summary['total_scraped']}</div>
                <div style="font-size:12px;color:#64748b;">Jobs Scraped</div>
              </td>
              <td style="padding:10px;text-align:center;background:#f1f5f9;">
                <div style="font-size:22px;font-weight:700;color:#10b981;">{summary['applied']}</div>
                <div style="font-size:12px;color:#64748b;">Applied</div>
              </td>
              <td style="padding:10px;text-align:center;background:#f1f5f9;">
                <div style="font-size:22px;font-weight:700;color:#f59e0b;">{summary['skipped']}</div>
                <div style="font-size:12px;color:#64748b;">Skipped</div>
              </td>
              <td style="padding:10px;text-align:center;background:#f1f5f9;">
                <div style="font-size:22px;font-weight:700;color:#ef4444;">{summary['expired']}</div>
                <div style="font-size:12px;color:#64748b;">No Longer Accepting</div>
              </td>
              <td style="padding:10px;text-align:center;background:#f1f5f9;border-radius:0 8px 8px 0;">
                <div style="font-size:22px;font-weight:700;color:#64748b;">{summary['not_applied']}</div>
                <div style="font-size:12px;color:#64748b;">Not Applied</div>
              </td>
            </tr>
          </table>

          <table style="width:100%;border-collapse:collapse;font-size:14px;">
            <thead>
              <tr style="background:#f8fafc;text-align:left;">
                <th style="padding:8px 12px;border-bottom:2px solid #e2e8f0;">Job Title</th>
                <th style="padding:8px 12px;border-bottom:2px solid #e2e8f0;">Company</th>
                <th style="padding:8px 12px;border-bottom:2px solid #e2e8f0;">Source</th>
                <th style="padding:8px 12px;border-bottom:2px solid #e2e8f0;">Status</th>
                <th style="padding:8px 12px;border-bottom:2px solid #e2e8f0;">Last Updated</th>
              </tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
      </div>
    </body>
    </html>
    """


def render_text(summary, rows):
    lines = [
        f"Job Bot Daily Report - {summary['date']} (PKT)",
        "",
        f"Scraped: {summary['total_scraped']} | Applied: {summary['applied']} | "
        f"Skipped: {summary['skipped']} | No Longer Accepting: {summary['expired']} | "
        f"Not Applied: {summary['not_applied']}",
        "",
    ]
    for r in rows:
        lines.append(f"- [{r['status']}] {r['job_title']} @ {r['company']} ({r['source']}) {_format_pkt(r['updated_at'])}")
    if not rows:
        lines.append("No jobs were scraped on this date.")
    return "\n".join(lines)


def send_report_email(summary, rows):
    if not (config.REPORT_SMTP_EMAIL and config.REPORT_SMTP_APP_PASSWORD and config.REPORT_RECIPIENT_EMAIL):
        raise RuntimeError(
            "Report email is not configured. Set REPORT_SMTP_EMAIL, REPORT_SMTP_APP_PASSWORD, "
            "REPORT_RECIPIENT_EMAIL in .env"
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Job Bot Daily Report — {summary['date']} ({summary['total_scraped']} scraped, {summary['applied']} applied)"
    msg["From"] = config.REPORT_SMTP_EMAIL
    msg["To"] = config.REPORT_RECIPIENT_EMAIL

    msg.attach(MIMEText(render_text(summary, rows), "plain"))
    msg.attach(MIMEText(render_html(summary, rows), "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
        server.login(config.REPORT_SMTP_EMAIL, config.REPORT_SMTP_APP_PASSWORD)
        server.send_message(msg)


def _already_sent_for(target_date: date) -> bool:
    data = _load_json(LAST_SENT_FILE, {})
    return data.get("date") == target_date.isoformat()


def _mark_sent(target_date: date):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LAST_SENT_FILE, "w", encoding="utf-8") as f:
        json.dump({"date": target_date.isoformat(), "sent_at": datetime.now(timezone.utc).isoformat()}, f)


def run(target_date: date, force: bool = False):
    if not force and _already_sent_for(target_date):
        log(f"Daily report for {target_date} already sent. Skipping.")
        return

    summary, rows = build_report(target_date)
    send_report_email(summary, rows)
    _mark_sent(target_date)
    log(f"Daily report for {target_date} sent: {summary}")


def main():
    parser = argparse.ArgumentParser(description="Send the Job Bot daily report email.")
    parser.add_argument("--force", action="store_true", help="Resend even if today's report was already sent (bypasses the once-per-day marker).")
    parser.add_argument("--date", type=str, default=None, help="Target report date (YYYY-MM-DD, PKT). Defaults to 'yesterday' in PKT.")
    args = parser.parse_args()

    now_pkt = datetime.now(PKT)
    target_date = date.fromisoformat(args.date) if args.date else (now_pkt.date() - timedelta(days=1))

    try:
        run(target_date, force=args.force)
    except Exception as exc:
        error(f"Daily report failed: {exc}")
        raise


if __name__ == "__main__":
    main()
