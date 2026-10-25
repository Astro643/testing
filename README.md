# AI ticket bot (multi-server)

## For you (the developer)
1. **Discord:** Developer Portal > New Application > Bot. Copy the token. Turn on **Message Content Intent**.
   Invite link scopes: `bot` + `applications.commands`. Permissions: Manage Channels, Manage Messages, View Channels, Send Messages, Read Message History, Attach Files, Embed Links.
2. **Upstash Vector:** create an index **with an embedding model** (e.g. `mixedbread-ai/mxbai-embed-large-v1`). Copy the REST URL and token.
3. **Upstash Redis:** create a database. Copy the REST URL and token. Do not enable eviction.
4. **AI (Cloudflare Workers AI):** create an API token with Workers AI permission and note your account ID. Set `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_KEY` and `CLOUDFLARE_MODEL` (one model, no backup). The free allowance is 10,000 Neurons a day for all servers combined; when it runs out, AI calls fail and tickets are escalated to staff. Requests are kept small: last 4 messages, 3+2 knowledge excerpts of up to 900 characters, 400 output tokens.
5. Fill `.env` (see `.env.example`), then `pip install -r requirements.txt` and `python bot.py`.
6. **Hosting:** run exactly one copy of the bot as a worker (no port, no volume needed). Put the variables in your host's secret settings, never in Git.

## For each server's admin
1. Invite the bot.
2. `/setup staff_role:@Staff log_channel:#ticket-logs` (optional `category:`; otherwise one is created).
3. Add knowledge: `/kb add` (type it in) or `/kb upload` (.txt / .md file). `/kb status` shows counts, `/kb clear confirm:True` wipes it.
4. `/ticketpanel` in the channel where the button should appear.
5. Make sure `@everyone` cannot see the ticket category.

## Behaviour (per server)
- Ticket numbers, knowledge base, learned answers and settings are separate for every server.
- Tickets 1-30 of a server: its staff role can see and talk; the AI answers the owner until staff speak; staff answers are saved as learned knowledge.
- Ticket 31+: staff role denied, any message from anyone but the owner is deleted (works on admins too). Administrators can still silently view the channel; Discord offers no way to stop that.
- If the AI can't solve it: it tells the user to DM staff and the ticket closes itself. If the user confirms it's fixed, it closes too.
- Transcripts go to the server's log channel when a ticket closes.
- Limits (per server): `DAILY_AI_LIMIT_PER_SERVER` AI replies per day, `MAX_KB_CHUNKS_PER_SERVER` knowledge entries, `MAX_USER_MESSAGES` per ticket.
- When the bot is removed from a server, that server's settings, counter, knowledge base and learned answers are deleted.
- `python ingest.py <guild_id> [--reset]` bulk-loads `kb/` files into one server (optional).
