#!/usr/bin/env python3
"""Gmail Auto-Labeler: classify inbox emails with Claude AI, then apply labels in bulk."""

import os
import sys
import json
import time
import base64
import threading
import webbrowser
import re

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import anthropic
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

app = Flask(__name__)
_gmail_service = None
_label_suggestions = []
_user_labels = []
_shutdown_event = threading.Event()


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


def _sender_domain(sender: str) -> str:
    m = re.search(r"@([\w.\-]+)", sender)
    return m.group(1).lower() if m else ""


_STOP_WORDS = {
    "the", "a", "an", "is", "in", "on", "at", "to", "for", "of",
    "and", "or", "your", "you", "re", "be", "has", "have", "with",
    "it", "this", "that", "we", "i", "my", "no", "subject",
}


def _subject_keywords(subject: str) -> list:
    words = re.findall(r"[a-z]+", subject.lower())
    return [w for w in words if w not in _STOP_WORDS][:8]


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


def extract_email_content(msg: dict) -> tuple:
    headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}
    sender = headers.get("from", "Unknown")
    subject = headers.get("subject", "(No Subject)")

    body = _extract_text(msg["payload"])
    if not body:
        body = _decode_b64(msg["payload"].get("body", {}).get("data", ""))

    body = " ".join(body.split())[:500]
    return sender, subject, body


def fetch_unlabeled_inbox(service, user_label_ids: set, max_results: int = 200) -> list:
    emails = []
    page_token = None

    while len(emails) < max_results:
        kwargs: dict = {
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
                userId="me", id=ref["id"], format="full"
            ).execute()
            if not (set(msg.get("labelIds", [])) & user_label_ids):
                emails.append(msg)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return emails


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


def classify_emails_with_claude(
    client: anthropic.Anthropic, labels: list, emails: list, history: list = None
) -> list:
    system_prompt = build_system_prompt(labels)
    history = history or []
    suggestions = []
    batch_size = 10
    total = len(emails)

    for batch_start in range(0, total, batch_size):
        batch = emails[batch_start : batch_start + batch_size]
        batch_end = min(batch_start + batch_size, total)
        print(f"  Classifying emails {batch_start + 1}–{batch_end} of {total} …")

        for email in batch:
            sender, subject, body = extract_email_content(email)

            examples_text = ""
            if history:
                relevant = get_relevant_examples(history, sender, subject)
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
                f"Classify this email:\n"
                f"From: {sender}\n"
                f"Subject: {subject}\n"
                f"Body preview: {body}"
                + examples_text
            )

            try:
                resp = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=300,
                    system=[
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_message}],
                )
                suggestion = parse_claude_response(resp.content[0].text)
            except Exception as exc:
                print(f"    Warning: Claude API error for {email['id']}: {exc}")
                suggestion = None

            suggestions.append({
                "id": email["id"],
                "sender": sender,
                "subject": subject,
                "suggestion": suggestion,
            })

        if batch_end < total:
            time.sleep(0.5)

    return suggestions


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "preview.html",
        suggestions=_label_suggestions,
        labels=_user_labels,
    )


@app.route("/apply", methods=["POST"])
def apply_labels():
    data = request.get_json(force=True)
    selected = data.get("selected", [])
    applied, errors = 0, 0
    history_entries = []

    suggestions_by_id = {s["id"]: s for s in _label_suggestions}

    for item in selected:
        try:
            for attempt in range(3):
                try:
                    _gmail_service.users().messages().modify(
                        userId="me",
                        id=item["id"],
                        body={"addLabelIds": [item["label_id"]]},
                    ).execute()
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1.5 ** attempt)
            applied += 1

            email_data = suggestions_by_id.get(item["id"], {})
            original_name = item.get("original_label_name", item.get("label_name", ""))
            applied_name = item.get("label_name", "")
            accepted = original_name == applied_name

            history_entries.append({
                "sender_domain": _sender_domain(email_data.get("sender", "")),
                "subject_keywords": _subject_keywords(email_data.get("subject", "")),
                "suggested_label_name": original_name,
                "applied_label_name": applied_name,
                "accepted": accepted,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
        except Exception as exc:
            print(f"Error labeling {item['id']}: {exc}")
            errors += 1

    if history_entries:
        save_history(history_entries)

    return jsonify({"applied": applied, "errors": errors, "total": len(selected)})


@app.route("/shutdown", methods=["POST"])
def shutdown():
    threading.Timer(0.3, _shutdown_event.set).start()
    return jsonify({"status": "ok"})


def _run_flask():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global _gmail_service, _label_suggestions, _user_labels

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: ANTHROPIC_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key."
        )

    print("Authenticating with Gmail …")
    _gmail_service = get_gmail_service()
    print("Authenticated.")

    print("\nFetching user-created labels …")
    labels = get_user_labels(_gmail_service)
    if not labels:
        sys.exit(
            "No user-created labels found.\n"
            "Create at least one label in Gmail first, then re-run."
        )
    _user_labels = labels
    print(f"Found {len(labels)} label(s): {', '.join(lb['name'] for lb in labels)}")

    print("\nFetching unlabeled inbox emails …")
    user_label_ids = {lb["id"] for lb in labels}
    emails = fetch_unlabeled_inbox(_gmail_service, user_label_ids)
    if not emails:
        sys.exit("No unlabeled inbox emails found. Nothing to do.")
    print(f"Found {len(emails)} unlabeled email(s) to process.")

    history = load_history()
    if history:
        print(f"Loaded {len(history)} past labeling decision(s) — will use as examples.")

    print("\nClassifying emails with Claude AI …")
    client = anthropic.Anthropic(api_key=api_key)
    _label_suggestions = classify_emails_with_claude(client, labels, emails, history=history)

    with_suggestions = sum(1 for s in _label_suggestions if s["suggestion"])
    print(
        f"\nDone. {with_suggestions}/{len(_label_suggestions)} emails received suggestions."
    )

    print("\nStarting preview server …")
    flask_thread = threading.Thread(target=_run_flask, daemon=True)
    flask_thread.start()

    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    print("Browser opening at http://127.0.0.1:5000  (Ctrl+C also quits)")

    try:
        _shutdown_event.wait()
    except KeyboardInterrupt:
        pass

    print("Goodbye!")


if __name__ == "__main__":
    main()
