"""DUALMIND Slack bot — scan your inbox before the AI reads it, from Slack.

Uses Slack **Socket Mode** (a websocket), so it runs straight from your laptop with
NO public URL / tunnel. Two triggers:

  * the slash command  /scan-inbox
  * a "Scan inbox" button on the bot's App Home tab

Both pull your UNREAD Gmail, run every message through DUALMIND, and post a Block
Kit report: cleared emails get an AI one-line summary; quarantined ones show only
the reason (the summarizer LLM never reads their content).

Setup (see integrations/README.md): create a Slack app with Socket Mode ON, scopes
chat:write + commands, a /scan-inbox slash command, then set:
  SLACK_BOT_TOKEN=xoxb-...   SLACK_APP_TOKEN=xapp-...
  GMAIL_USER=you@gmail.com   GMAIL_APP_PASSWORD=...(16-char app password)
Run:  python integrations/slack_bot.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_and_seed
from integrations.email_guard import scan_gmail_inbox

SCAN_LIMIT = int(os.environ.get("DUALMIND_SCAN_LIMIT", "10"))
CFG = load_and_seed()


def _report_blocks(rep: dict) -> list:
    head = (f":shield: *DUALMIND inbox scan* — {rep['scanned']} scanned · "
            f"*{rep['n_cleared']} cleared* · :no_entry: *{rep['n_quarantined']} quarantined*"
            + ("   _(mock mode — set ANTHROPIC_API_KEY for real LLMs)_"
               if rep.get("mode") == "mock" else ""))
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": head}},
              {"type": "divider"}]
    if rep["n_quarantined"]:
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
                       "text": ":no_entry: *Quarantined — the AI never read these:*"}})
        for q in rep["quarantined"]:
            blocks.append({"type": "section", "text": {"type": "mrkdwn",
                "text": (f"*{_esc(q['subject']) or '(no subject)'}*\n"
                         f"from `{_esc(q['sender'])}`\n"
                         f":warning: caught by *{q['caught_by']}* · "
                         f"verdict *{q['verdict']}* · risk {q['risk']}\n_{_esc(q['reason'])}_")}})
    if rep["n_cleared"]:
        blocks.append({"type": "divider"})
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
                       "text": ":white_check_mark: *Cleared — safe for the AI to read:*"}})
        for c in rep["cleared"]:
            summ = _esc(c.get("summary") or "(no summary)")
            blocks.append({"type": "section", "text": {"type": "mrkdwn",
                "text": (f"*{_esc(c['subject']) or '(no subject)'}*\n"
                         f"from `{_esc(c['sender'])}`\n:speech_balloon: {summ}")}})
    if not rep["scanned"]:
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
                       "text": ":inbox_tray: No unread email to scan — inbox is clear."}})
    return blocks


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")[:600]


def _home_view() -> dict:
    return {"type": "home", "blocks": [
        {"type": "header", "text": {"type": "plain_text", "text": "🛡️ DUALMIND email guard"}},
        {"type": "section", "text": {"type": "mrkdwn", "text":
            "Scan your *unread* Gmail through DUALMIND *before* any assistant AI reads it. "
            "Malicious/indirect-injection emails are quarantined; only cleared mail is summarized."}},
        {"type": "actions", "elements": [
            {"type": "button", "style": "primary", "action_id": "scan_inbox_btn",
             "text": {"type": "plain_text", "text": "🛡️ Scan inbox now"}}]},
        {"type": "context", "elements": [{"type": "mrkdwn", "text":
            "Or run `/scan-inbox` in any channel or DM."}]}]}


def main():
    bot = os.environ.get("SLACK_BOT_TOKEN")
    app_token = os.environ.get("SLACK_APP_TOKEN")
    if not bot or not app_token:
        print("Set SLACK_BOT_TOKEN (xoxb-...) and SLACK_APP_TOKEN (xapp-...). "
              "See integrations/README.md.")
        sys.exit(1)
    try:
        from slack_bolt import App
        from slack_bolt.adapter.socket_mode import SocketModeHandler
    except ImportError:
        print("pip install slack_bolt   (in the .venv)")
        sys.exit(1)

    app = App(token=bot)

    @app.command("/scan-inbox")
    def _cmd(ack, respond):
        ack(":shield: Scanning your inbox through DUALMIND… (a few seconds)")
        try:
            rep = scan_gmail_inbox(CFG, limit=SCAN_LIMIT)
            respond(blocks=_report_blocks(rep), response_type="ephemeral")
        except Exception as e:
            respond(f":x: Scan failed: {e}")

    @app.event("app_home_opened")
    def _home(client, event):
        client.views_publish(user_id=event["user"], view=_home_view())

    @app.action("scan_inbox_btn")
    def _btn(ack, body, client):
        ack()
        uid = body["user"]["id"]
        client.chat_postMessage(channel=uid, text=":shield: Scanning your inbox…")
        try:
            rep = scan_gmail_inbox(CFG, limit=SCAN_LIMIT)
            client.chat_postMessage(channel=uid, blocks=_report_blocks(rep),
                                    text="DUALMIND inbox scan complete")
        except Exception as e:
            client.chat_postMessage(channel=uid, text=f":x: Scan failed: {e}")

    print(f"\n  🛡️ DUALMIND Slack bot (Socket Mode) — provider scan limit={SCAN_LIMIT}")
    print("  Open the app's Home tab and click 'Scan inbox now', or run /scan-inbox\n")
    SocketModeHandler(app, app_token).start()


if __name__ == "__main__":
    main()
