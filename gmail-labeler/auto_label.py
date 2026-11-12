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

import anthropic
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


CLAUDE_CONFIDENCE_THRESHOLD = 90

_STOP_WORDS = {
    "the", "a", "an", "is", "in", "on", "at", "to", "for", "of",
    "and", "or", "your", "you", "re", "be", "has", "have", "with",
    "it", "this", "that", "we", "i", "my", "no", "subject",
}


def _sender_domain(sender: str) -> str:
    m = re.search(r"@([\w.\-]+)", sender)
    return m.group(1).lower() if m else ""


def _subject_keywords(subject: str) -> list:
    words = re.findall(r"[a-z]+", subject.lower())
    return [w for w in words if w not in _STOP_WORDS][:8]


def _get_header(msg: dict, name: str) -> str:
    for h in msg.get("payload", {}).get("headers", []):
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def _decode_b64(data: str) -> str:
    if not data:
        return ""
    try:
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_text(part: dict) -> str:
    if part.get("mimeType") == "text/plain":
        return _decode_b64(part.get("body", {}).get("data", ""))
    for sub in part.get("parts", []):
        text = _extract_text(sub)
        if text:
            return text
    return ""


def fetch_full_body(service, msg_id: str) -> str:
    msg = service.users().messages().get(
        userId="me", id=msg_id, format="full"
    ).execute()
    body = _extract_text(msg["payload"])
    if not body:
        body = _decode_b64(msg["payload"].get("body", {}).get("data", ""))
    return " ".join(body.split())[:500]


# ── History helpers ───────────────────────────────────────────────────────────

def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_history(new_entries: list) -> None:
    history = load_history()
    history.extend(new_entries)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def get_relevant_examples(history: list, sender: str, subject: str, n: int = 5) -> list:
    domain = _sender_domain(sender)
    keywords = set(_subject_keywords(subject))
    scored = []
    for entry in history:
        score = 0
        if domain and entry.get("sender_domain") == domain:
            score += 3
        score += len(keywords & set(entry.get("subject_keywords", [])))
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:n]]


# ── Claude helpers ────────────────────────────────────────────────────────────

def build_system_prompt(labels: list) -> str:
    label_json = json.dumps(
        [{"id": lb["id"], "name": lb["name"]} for lb in labels], indent=2
    )
    return (
        "You are an email labeling assistant.\n"
        "Your job: pick the single most appropriate Gmail label for an email based on its content.\n\n"
        f"Available labels:\n{label_json}\n\n"
        "Rules:\n"
        "- Respond with ONLY a JSON object — no markdown, no code fences.\n"
        '- If a label clearly fits, return:\n'
        '  {"label_id":"<id>","label_name":"<name>","confidence":<0-100>,"reasoning":"<one sentence>"}\n'
        "- If no label fits well, return exactly: null\n\n"
        "When past decisions are provided, use them to align with the user's labeling preferences."
    )


def parse_claude_response(text: str):
    text = text.strip()
    if text.lower() == "null":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def classify_with_claude(client, system_prompt: str, labels: list, history: list,
                         sender: str, subject: str, body: str):
    relevant = get_relevant_examples(history, sender, subject)
    examples_text = ""
    if relevant:
        lines = []
        for ex in relevant:
            if ex["accepted"]:
                action = f"labeled '{ex['applied_label_name']}' (accepted)"
            else:
                action = (
                    f"suggested '{ex['suggested_label_name']}' "
                    f"but user corrected to '{ex['applied_label_name']}'"
                )
            lines.append(f"  - @{ex['sender_domain']}: {action}")
        examples_text = "\n\nRelevant past decisions:\n" + "\n".join(lines)

    user_message = (
        f"Classify this email:\nFrom: {sender}\nSubject: {subject}\nBody preview: {body}"
        + examples_text
    )

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_message}],
    )
    return parse_claude_response(resp.content[0].text)


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
    print(f"=== Run started: {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

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

    api_key = os.getenv("ANTHROPIC_API_KEY")
    claude_client = anthropic.Anthropic(api_key=api_key) if api_key else None
    if claude_client:
        print(f"Claude enabled — will auto-label unknown senders with confidence ≥ {CLAUDE_CONFIDENCE_THRESHOLD}%")
    else:
        print("ANTHROPIC_API_KEY not set — skipping Claude fallback for unknown senders.")

    print("\nAuthenticating with Gmail …")
    service = get_gmail_service()
    print("Authenticated.")

    labels = get_user_labels(service)
    label_name_to_id = {lb["name"]: lb["id"] for lb in labels}
    label_id_to_name = {lb["id"]: lb["name"] for lb in labels}
    user_label_ids = {lb["id"] for lb in labels}

    system_prompt = build_system_prompt(labels) if claude_client else None

    print("\nFetching unlabeled inbox emails …")
    emails = fetch_unlabeled_inbox(service, user_label_ids)
    print(f"Found {len(emails)} unlabeled email(s).")

    applied = skipped_unknown = skipped_no_label = errors = 0
    history_entries = []

    for msg in emails:
        sender  = _get_header(msg, "From")
        subject = _get_header(msg, "Subject") or "(No Subject)"
        domain  = _sender_domain(sender)

        if domain not in rules:
            if claude_client:
                try:
                    body = fetch_full_body(service, msg["id"])
                    suggestion = classify_with_claude(
                        claude_client, system_prompt, labels, history,
                        sender, subject, body
                    )
                    confidence = suggestion.get("confidence", 0) if suggestion else 0
                    if suggestion and confidence >= CLAUDE_CONFIDENCE_THRESHOLD:
                        label_id = suggestion.get("label_id")
                        label_name = suggestion.get("label_name", "")
                        if not label_id or label_id not in label_id_to_name:
                            skipped_unknown += 1
                            continue
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
                        print(f"  ~ @{domain}: \"{subject[:60]}\"  →  {label_name} (Claude {confidence}%)")
                        applied += 1
                        history_entries.append({
                            "sender_domain": domain,
                            "subject_keywords": _subject_keywords(subject),
                            "suggested_label_name": label_name,
                            "applied_label_name": label_name,
                            "accepted": True,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        })
                    else:
                        skipped_unknown += 1
                except Exception as exc:
                    print(f"  ! Claude error for {msg['id']}: {exc}")
                    skipped_unknown += 1
            else:
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

    if history_entries:
        save_history(history_entries)

    print(
        f"\nDone. {applied} labeled automatically, "
        f"{skipped_unknown} skipped (unknown domain), "
        f"{errors} errors."
    )
    if skipped_unknown:
        print("Run label_emails.py to review and label the unknown senders.")
    print(f"=== Run finished: {time.strftime('%Y-%m-%d %H:%M:%S')} ===")


if __name__ == "__main__":
    main()
