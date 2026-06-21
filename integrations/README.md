# DUALMIND email guard — Slack + website

Check your **real** email through DUALMIND **before any assistant AI reads it**.
Pulls unread Gmail (read-only — uses `BODY.PEEK`, so nothing is marked read), runs
each message through the full pipeline, and **gates**: only *cleared* mail is handed
to the AI summarizer; *quarantined* mail (block/review) is never sent to the LLM.

> DUALMIND's internal Reader is the sandboxed, no-privilege model that's *designed*
> to read untrusted text safely. This gate guarantees your **privileged**
> assistant/summarizer only ever reads emails DUALMIND cleared.

## 1. Gmail (read-only) — ~3 min
1. Enable 2-Step Verification on your Google account.
2. Create an **App Password**: Google Account → Security → App passwords → "Mail" →
   copy the 16-character password.
3. Put it in `.env` (gitignored):
   ```
   GMAIL_USER=you@gmail.com
   GMAIL_APP_PASSWORD=abcd efgh ijkl mnop      # the 16-char app password
   ANTHROPIC_API_KEY=sk-ant-...                # so the summarizer + DUALMIND use real Claude
   ```

Test the pipe alone:
```bash
python -c "from config import load_and_seed; from integrations.email_guard import scan_gmail_inbox; import json; print(json.dumps(scan_gmail_inbox(load_and_seed(), limit=5), indent=2))"
```

## 2a. Run it from the website (button)
```bash
python demo/web_app.py     # http://127.0.0.1:8123
```
Click **"📥 Scan my inbox (before AI reads)"** in the header. Quarantined emails show
the catching layer + reason; cleared emails show an AI one-line summary.

## 2b. Run it from Slack (button + slash command) — ~5 min
Slack uses **Socket Mode**, so no public URL / tunnel is needed.

1. https://api.slack.com/apps → **Create New App** → *From scratch*.
2. **Socket Mode** → toggle **On** → generate an **App-Level Token** with scope
   `connections:write` → copy `xapp-...`.
3. **OAuth & Permissions** → **Bot Token Scopes**: add `chat:write` and `commands`.
4. **Slash Commands** → **Create New Command**: `/scan-inbox` (any description;
   Request URL is ignored in Socket Mode).
5. **App Home** → enable the **Home Tab** (and "Allow users to send messages").
6. **Event Subscriptions** → On → subscribe to bot event `app_home_opened`.
7. **Install App** to your workspace → copy the **Bot User OAuth Token** `xoxb-...`.
8. Add to `.env`:
   ```
   SLACK_BOT_TOKEN=xoxb-...
   SLACK_APP_TOKEN=xapp-...
   ```
9. Install the dep and run:
   ```bash
   pip install slack_bolt
   python integrations/slack_bot.py
   ```
Open the bot's **Home** tab → **🛡️ Scan inbox now**, or type **`/scan-inbox`** anywhere.

## Notes
- `DUALMIND_SCAN_LIMIT` (default 10) caps how many unread emails a scan processes.
- Without `ANTHROPIC_API_KEY` the pipeline runs in **mock** mode (deterministic
  stand-ins) — wiring still works, but use a real key to see real detection/summaries.
- Read-only: nothing is sent, deleted, or marked read. Quarantine is a report only.
