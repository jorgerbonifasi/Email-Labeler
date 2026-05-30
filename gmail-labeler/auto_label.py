#!/usr/bin/env python3
"""Auto-label Gmail inbox emails based on past decisions in history.json.

Domains with a clear majority label are applied automatically.
Unknown domains and tied domains are left for manual review via label_emails.py.
"""

import os
import sys
import re
import time
import json
import base64
from collections import Counter

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
HISTORY_FILE = "history.json"

SYSTEM_LABEL_IDS = {
    "INBOX", "SENT", "DRAFTS", "SPAM", "TRASH", "STARRED", "IMPORTANT",
    "UNREAD", "CHAT", "CATEGORY_PERSONAL", "CATEGORY_SOCIAL",
    "CATEGORY_PROMOTIONS", "CATEGORY_UPDATES", "CATEGORY_FORUMS",
}


# ── Gmail helpers ─────────────────────────────────────────────────────────────

def get_gmail_service():
    creds = None
    token_path = "token.json"
    creds_path = "credentials.json"

    if not os.path.exists(creds_path):
        sys.exit(
            "ERROR: 'credentials.json' not found.\n"
            "Download it from Google Cloud Console → APIs & Services → Credentials."
        )

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as fh:
            fh.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def get_user_labels(service):
    result = service.users().labels().list(userId="me").execute()
    return [
        lb for lb in result.get("labels", [])
        if lb["id"] not in SYSTEM_LABEL_IDS
        and not lb["id"].startswith("CATEGORY_")
    ]


def fetch_unlabeled_inbox(service, user_label_ids: set, max_results: int = 200) -> list:
    emails = []
    page_token = None

    while len(emails) < max_results:
        kwargs = {
            "userId": "me",
            "labelIds": ["INBOX"],
            "maxResults": min(100, max_results - len(emails) + 50),
        }
        if page_token:
            kwargs["pageToken"] = page_token

        resp = service.users().messages().list(**kwargs).execute()
        messages = resp.get("messages", [])
        if not messages:
            break

        for ref in messages:
            if len(emails) >= max_results:
                break
            msg = service.users().messages().get(
                userId="me", id=ref["id"], format="metadata",
                metadataHeaders=["From", "Subject"],
            ).execute()
            if not (set(msg.get("labelIds", [])) & user_label_ids):
                emails.append(msg)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return emails


def _sender_domain(sender: str) -> str:
    m = re.search(r"@([\w.\-]+)", sender)
    return m.group(1).lower() if m else ""


def _get_header(msg: dict, name: str) -> str:
    for h in msg.get("payload", {}).get("headers", []):
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


# ── History helpers ───────────────────────────────────────────────────────────

def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def build_domain_rules(history: list) -> dict:
    """Return {domain: label_name} for domains with a strict majority label (>50%).

    With only 1 decision it's 100% — qualifies.
    With 2 conflicting decisions it's 50/50 — skipped.
    With 3 decisions where 2 agree it's 66% — qualifies.
    """
    domain_counts: dict[str, Counter] = {}
    for entry in history:
        domain = entry.get("sender_domain", "")
        label = entry.get("applied_label_name", "")
        if domain and label:
            domain_counts.setdefault(domain, Counter())[label] += 1

    rules = {}
    for domain, counter in domain_counts.items():
        total = sum(counter.values())
        best_label, top_count = counter.most_common(1)[0]
        if top_count / total > 0.5:
            rules[domain] = best_label
    return rules


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    history = load_history()
    if not history:
        sys.exit(
            "No history found. Run label_emails.py first and apply some labels "
            "to build history.json."
        )

    rules = build_domain_rules(history)
    if not rules:
        sys.exit(
            "No domains with a clear majority label yet.\n"
            "Keep using label_emails.py — more decisions will resolve ties."
        )

    print(f"Auto-label rules ({len(rules)} domain(s)):")
    for domain, label in sorted(rules.items()):
        print(f"  @{domain}  →  {label}")

    print("\nAuthenticating with Gmail …")
    service = get_gmail_service()
    print("Authenticated.")

    labels = get_user_labels(service)
    label_name_to_id = {lb["name"]: lb["id"] for lb in labels}
    user_label_ids = {lb["id"] for lb in labels}

    print("\nFetching unlabeled inbox emails …")
    emails = fetch_unlabeled_inbox(service, user_label_ids)
    print(f"Found {len(emails)} unlabeled email(s).")

    applied = skipped_unknown = skipped_no_label = errors = 0

    for msg in emails:
        sender  = _get_header(msg, "From")
        subject = _get_header(msg, "Subject") or "(No Subject)"
        domain  = _sender_domain(sender)

        if domain not in rules:
            skipped_unknown += 1
            continue

        label_name = rules[domain]
        label_id   = label_name_to_id.get(label_name)

        if not label_id:
            print(f"  Warning: label '{label_name}' not found in Gmail — skipping @{domain}")
            skipped_no_label += 1
            continue

        try:
            for attempt in range(3):
                try:
                    service.users().messages().modify(
                        userId="me",
                        id=msg["id"],
                        body={"addLabelIds": [label_id]},
                    ).execute()
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1.5 ** attempt)

            print(f"  + @{domain}: \"{subject[:60]}\"  →  {label_name}")
            applied += 1
        except Exception as exc:
            print(f"  ! Error labeling {msg['id']}: {exc}")
            errors += 1

    print(
        f"\nDone. {applied} labeled automatically, "
        f"{skipped_unknown} skipped (unknown domain), "
        f"{errors} errors."
    )
    if skipped_unknown:
        print("Run label_emails.py to review and label the unknown senders.")


if __name__ == "__main__":
    main()
