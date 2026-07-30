# Security & data-handling memo (for IT review)

**One sentence:** all financial data, all AI models, and all processing stay
on Kronospan-controlled hardware; nothing is sent to any cloud or third-party
service.

## Architecture
- Models run locally via Ollama (DGX Spark). No internet access is required
  or used at answer time.
- Data (Excel/PDF exports) is indexed into `<RAG_DATA_DIR>` (Docker volume
  `/data`). Raw files, the index, embeddings and caches never leave the host.
- The optional backend API (`jarvisman.server`) binds to the LAN; front-ends
  (desktop app / browser page) talk to it over JSON.

## Access control
- Set `RAG_API_KEY=<secret>` on the api service; every endpoint except the
  web page shell and `/health` then requires the `X-Api-Key` header.
- For encryption in transit, terminate TLS at a reverse proxy in front of
  port 8800 (nginx/caddy/traefik — see DEPLOYMENT.md). Do not expose 8800
  beyond the office network.

## Audit trail
- Every answered question is appended to
  `<RAG_DATA_DIR>/audit/audit-YYYYMM.jsonl`: timestamp, client, question,
  tool, generated code, tables used, sources, answer excerpt, duration.
  This is the question → code → tables → answer chain an auditor can replay.
- Retention: the files are plain JSONL per month — archive or purge per
  Kronospan's document-retention policy.

## Backup / recovery
- Back up the `/data` volume (index, rules.txt, scoreboard, audit logs) and
  the source export folder. Rebuilding from scratch = re-run indexing on the
  exports; nothing else is stateful.

## Verification
- Deterministic data checks (`python -m eval.run_use_cases`) and graded
  answer checks (`python test_queries.py`) write their latest results to the
  System page scoreboard, so accuracy is continuously visible, not assumed.
