"""DUALMIND email guard — check email BEFORE the assistant AI ever reads it.

Pulls UNREAD email over IMAP using ``BODY.PEEK`` (so nothing is marked read), runs
every message through the full DUALMIND pipeline, and GATES the result:

  * verdict "allow"  -> CLEARED   -> handed to the AI summarizer.
  * block / review   -> QUARANTINED -> the summarizer LLM NEVER sees its content.

DUALMIND's own Reader (System 1) is the sandboxed, no-privilege model that is
*designed* to read untrusted text safely; this gate guarantees your *privileged*
assistant/summarizer only ever reads emails DUALMIND cleared. Pure logic
(``guard_emails``) is separated from IMAP I/O (``fetch_unread_gmail``) so it can be
tested with fakes and reused by both the Slack bot and the web app.
"""

from __future__ import annotations

import email
import imaplib
import os
from dataclasses import dataclass, field
from email.header import decode_header, make_header
from typing import Callable, List, Optional

DEFAULT_GOAL = "Triage and summarize my new email."
SUMMARY_MODEL_KEY = "reader"  # use the cheaper model for summaries


@dataclass
class Email:
    sender: str
    subject: str
    body: str
    uid: str = ""

    @property
    def content(self) -> str:
        # Same shape the benchmark scores real LLMail-Inject mail in, so DUALMIND
        # behaves identically here.
        return f"Subject of the email: {self.subject}.   Body: {self.body}"


@dataclass
class GuardReport:
    scanned: int = 0
    cleared: List[dict] = field(default_factory=list)      # {sender,subject,risk,summary}
    quarantined: List[dict] = field(default_factory=list)  # {sender,subject,verdict,caught_by,risk,reason}
    mode: str = "live"

    def as_dict(self) -> dict:
        return {"scanned": self.scanned, "n_cleared": len(self.cleared),
                "n_quarantined": len(self.quarantined), "mode": self.mode,
                "cleared": self.cleared, "quarantined": self.quarantined}


# --------------------------------------------------------------------------- #
# IMAP intake (side-effecting). BODY.PEEK => we DON'T mark mail read.
# --------------------------------------------------------------------------- #
def _decode(s) -> str:
    try:
        return str(make_header(decode_header(s or "")))
    except Exception:
        return s or ""


def _extract_body(msg) -> str:
    if msg.is_multipart():
        # prefer text/plain, skip attachments
        for part in msg.walk():
            if (part.get_content_type() == "text/plain"
                    and "attachment" not in str(part.get("Content-Disposition", ""))):
                try:
                    return part.get_payload(decode=True).decode(
                        part.get_content_charset() or "utf-8", "replace")
                except Exception:
                    continue
        # fall back to stripped html
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                import re
                try:
                    html = part.get_payload(decode=True).decode(
                        part.get_content_charset() or "utf-8", "replace")
                    return re.sub(r"<[^>]+>", " ", html)
                except Exception:
                    continue
        return ""
    try:
        return msg.get_payload(decode=True).decode(
            msg.get_content_charset() or "utf-8", "replace")
    except Exception:
        return str(msg.get_payload())


def fetch_unread_gmail(user: Optional[str] = None, app_password: Optional[str] = None,
                       limit: int = 15, host: str = "imap.gmail.com") -> List[Email]:
    """Fetch up to ``limit`` UNREAD messages without marking them read (BODY.PEEK)."""
    user = user or os.environ.get("GMAIL_USER")
    app_password = app_password or os.environ.get("GMAIL_APP_PASSWORD")
    if not user or not app_password:
        raise RuntimeError("Set GMAIL_USER and GMAIL_APP_PASSWORD (Gmail app password).")
    conn = imaplib.IMAP4_SSL(host)
    try:
        conn.login(user, app_password)
        conn.select("INBOX")
        typ, data = conn.search(None, "UNSEEN")
        ids = data[0].split()
        ids = ids[-limit:] if limit else ids
        out: List[Email] = []
        for i in reversed(ids):  # newest first
            typ, msg_data = conn.fetch(i, "(BODY.PEEK[])")  # PEEK = stays unread
            if typ != "OK" or not msg_data or not msg_data[0]:
                continue
            msg = email.message_from_bytes(msg_data[0][1])
            out.append(Email(sender=_decode(msg.get("From")),
                             subject=_decode(msg.get("Subject")),
                             body=_extract_body(msg).strip(),
                             uid=i.decode() if isinstance(i, bytes) else str(i)))
        return out
    finally:
        try:
            conn.logout()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# The gate (pure-ish; dm + summarize injected => testable with fakes).
# --------------------------------------------------------------------------- #
def guard_emails(emails: List[Email], dm, user_goal: str = DEFAULT_GOAL,
                 summarize: Optional[Callable[[Email], str]] = None,
                 mode: str = "live", quarantine_threshold: float = 0.9) -> GuardReport:
    """Run each email through DUALMIND; summarize ONLY the cleared ones.

    Gate policy for email triage: quarantine on a real INJECTION signal — a tainted
    action (the email tried to steer a tool/action) or risk >= quarantine_threshold —
    not on every moderate-risk message. Summarizing is read-only, so ordinary mail
    with calls-to-action stays cleared. This is a product operating point on top of
    DUALMIND; it does NOT change the benchmark's stricter block threshold.
    """
    rep = GuardReport(scanned=len(emails), mode=mode)
    for em in emails:
        # Fail closed: any error analyzing an email -> quarantine it (the AI must not
        # read something we couldn't vet), never crash the whole scan.
        try:
            tr = dm.process(em.content, user_goal=user_goal, content_type="email")
        except Exception as e:  # pragma: no cover - pipeline/LLM dependent
            rep.quarantined.append({
                "sender": em.sender, "subject": em.subject, "verdict": "block",
                "caught_by": "error", "risk": 1.0,
                "reason": f"could not analyze — quarantined fail-closed ({e})"})
            continue
        risk = round(float(getattr(tr, "risk", 0.0)), 3)
        findings = getattr(tr, "taint_findings", None) or []
        pf, rd = getattr(tr, "prefilter", None), getattr(tr, "reader", None)
        # Quarantine on an injection-SPECIFIC signal, not the noisy aggregate risk
        # (ordinary mail with calls-to-action scores ~0.85 too). The separators that
        # actually fire only on attacks: a tainted action, the Reader judging the
        # content an injection attempt, or the pre-filter matching injection patterns.
        # Only a high-confidence pre-filter BLOCK counts — "near_miss" fires on
        # ordinary rich-formatted mail (smart quotes, em-dashes, zero-width tracking
        # chars look like homoglyph/zero-width obfuscation). The Reader catches the
        # subtle injections those near-misses would otherwise over-flag.
        prefilter_flag = bool(pf) and getattr(pf, "verdict", "pass") == "block"
        reader_flag = bool(rd) and getattr(rd, "suspicious", False)
        is_threat = (bool(findings) or reader_flag or prefilter_flag
                     or risk >= quarantine_threshold)
        if not is_threat:
            summary = ""
            if summarize is not None:
                try:
                    summary = summarize(em)
                except Exception as e:  # pragma: no cover - summarizer/LLM dependent
                    summary = f"(summary unavailable: {e})"
            rep.cleared.append({"sender": em.sender, "subject": em.subject,
                                "risk": risk, "summary": summary})
        else:
            if findings:
                reason, caught = findings[0], "taint"
            elif reader_flag:
                reason = "Reader flagged an injection attempt: " + (
                    (rd.intent.get("suspicion_reason") if getattr(rd, "intent", None) else "")
                    or "content tries to override instructions")
                caught = "dual_llm"
            elif prefilter_flag:
                reason = "pre-filter matched injection patterns: " + ", ".join(
                    (pf.signals or [])[:4])
                caught = "prefilter"
            else:
                reason = f"high injection risk {risk}"
                caught = getattr(tr, "caught_by", "?")
            rep.quarantined.append({
                "sender": em.sender, "subject": em.subject, "verdict": "block",
                "caught_by": caught, "risk": risk, "reason": reason})
    return rep


# --------------------------------------------------------------------------- #
# Summarizer — the PRIVILEGED "AI that reads your email". Only called on cleared.
# --------------------------------------------------------------------------- #
def make_summarizer(provider, model: str):
    def _summarize(em: Email) -> str:
        prompt = (f"Summarize this email in one sentence for a busy reader.\n\n"
                  f"From: {em.sender}\nSubject: {em.subject}\n\n{em.body[:4000]}")
        return provider.complete(prompt, model=model, max_tokens=120,
                                 label="email_summary").text.strip()
    return _summarize


# --------------------------------------------------------------------------- #
# Convenience: build DUALMIND, fetch real unread mail, gate it, return a report.
# --------------------------------------------------------------------------- #
def scan_gmail_inbox(cfg, limit: int = 15, summarize_cleared: bool = True) -> dict:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from llm import get_provider
    from pipeline import DualMind
    from prefilter import PreFilter

    provider = get_provider(cfg)
    mode = provider.name
    if mode == "mock":
        from eval.mock_agents import install_mock_agents
        install_mock_agents(provider)
    prefilter = PreFilter.with_default_classifier(cfg)
    dm = DualMind(cfg, provider=provider, prefilter=prefilter)

    emails = fetch_unread_gmail(limit=limit)
    summarize = None
    if summarize_cleared:
        model = cfg.get("models", {}).get(SUMMARY_MODEL_KEY, "claude-haiku-4-5")
        summarize = make_summarizer(provider, model)
    return guard_emails(emails, dm, summarize=summarize, mode=mode).as_dict()
