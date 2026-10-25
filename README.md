# AI ticket bot (multi-server)

## For you (the developer)
1. **Discord:** Developer Portal > New Application > Bot. Copy the token. Turn on **Message Content Intent**.
   Invite link scopes: `bot` + `applications.commands`. Permissions: Manage Channels, Manage Messages, View Channels, Send Messages, Read Message History, Attach Files, Embed Links.
2. **Upstash Vector:** create an index **with an embedding model** (Upstash currently offers `openai/text-embedding-3-small` for dense indexes; choose it). Copy the REST URL and token.
3. **Upstash Redis:** create a database. Copy the REST URL and token. Do not enable eviction.
4. **AI (Google AI Studio / Gemini):** create an API key at aistudio.google.com ("Get API key") and set `GEMINI_API_KEY`. The model is `GEMINI_MODEL` (default `gemini-3.5-flash-lite`; one model, no backup). The free tier has its own per-minute and per-day limits shared by all servers; when they are hit, AI calls fail and tickets are escalated. Note that Google may use free-tier content to improve its products (the Privacy Policy says so); enabling billing on the key's Google Cloud project switches to the paid terms.
5. Fill `.env` (see `.env.example`), then `pip install -r requirements.txt` and `python bot.py`.
6. **Hosting:** run exactly one copy of the bot as a worker (no port, no volume needed). Put the variables in your host's secret settings, never in Git.

## For each server's admin
1. Invite the bot.
2. `/setup log_channel:#ticket-logs` (optional `staff_role:` to keep a role out of tickets, optional `category:`; otherwise one is created).
3. Optional: add details about your own server with `/kb add` (type it in) or `/kb upload` (.txt / .md file). `/kb status` shows the count, `/kb clear confirm:True` wipes it. If `/kb add` fails, the message shows the real error.
4. `/ticketpanel` in the channel where the button should appear.
5. Make sure `@everyone` cannot see the ticket category.

## Behaviour (per server)
- Every ticket is AI-only from ticket 1: anyone but the ticket owner has their messages deleted (works on admins too). Administrators can still silently view the channel; Discord offers no way to stop that.
- The AI answers general Discord questions from its own knowledge. Questions about the specific server are answered only from the knowledge base; if the details aren't there, it escalates.
- If the AI can't solve it: it tells the user to DM staff and the ticket closes itself. If the user confirms it's fixed, it closes too.
- Transcripts go to the server's log channel when a ticket closes.
- Limits (per server): `DAILY_AI_LIMIT_PER_SERVER` AI replies per day, `MAX_KB_CHUNKS_PER_SERVER` knowledge entries, `MAX_USER_MESSAGES` per ticket.
- When the bot is removed from a server, that server's settings, counter and knowledge base are deleted.
- `python ingest.py <guild_id> [--reset]` bulk-loads `kb/` files into one server (optional).
