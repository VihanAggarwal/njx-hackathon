"""DUALMIND integrations — gate real email through DUALMIND before an AI reads it."""

from .email_guard import fetch_unread_gmail, guard_emails, scan_gmail_inbox

__all__ = ["fetch_unread_gmail", "guard_emails", "scan_gmail_inbox"]
