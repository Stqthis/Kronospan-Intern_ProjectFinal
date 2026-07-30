# Deployment

## Local (no Docker)

    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    ollama serve &                       # plus: ollama pull qwen2.5-coder:32b
    python app.py                        # desktop GUI
    # or headless:
    python -m jarvisman.cli index ./my_files
    python -m jarvisman.cli chat

## Docker (headless)

    docker compose up -d ollama
    docker compose exec ollama ollama pull qwen2.5-coder:32b
    docker compose exec ollama ollama pull nomic-embed-text
    mkdir -p files                       # drop your PDFs/Excels here
    docker compose run --rm app index /files
    docker compose run --rm app ask "Total funds of Lignum Technologies AG in EUR?"
    docker compose run --rm app chat

The index persists in the `jarvis_index` volume; re-running `index` on new
files UPDATES the collection (previously indexed files are kept, re-indexed
files replace their old version). GUI and CLI share the same index format.

## Docker (GUI, Linux/X11)

    xhost +local:docker
    docker compose run --rm \
      -e DISPLAY=$DISPLAY -e QT_QPA_PLATFORM=xcb \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      --entrypoint python app app.py

On Windows/macOS use an X server (VcXsrv / XQuartz) and set DISPLAY
accordingly, or simply run the GUI natively and let only Ollama run in Docker
(`OLLAMA_HOST=http://localhost:11434` is the default).

## Scaling to many files

- Index in batches: `python -m jarvisman.cli index folderA folderB ...` --
  updates are incremental, and per-table LLM work (profiling, cards) is
  cached by schema hash, so unchanged files cost zero model calls on rebuild.
- Verify the foundation any time: `python -m jarvisman.cli check --data <folder>`
  runs the ground-truth use-case checks offline.
- For lower latency, route planning to a smaller model:
  `RAG_PLAN_MODEL=qwen2.5-coder:7b` (keep the 32b for synthesis).


## Backend API (Docker) — the model as a service

Run the whole pipeline (index + agent + Ollama access) as a persistent
container and let any front-end connect to it:

    docker compose up -d ollama api
    curl http://localhost:8800/health        # {"ok": true, ...}
    curl -X POST http://localhost:8800/ask -d '{"q": "total LTL outstanding per lender"}'

The `api` service mounts the same index volume as the CLI (`/data`), so the
existing index is served as-is — nothing is re-indexed.

Desktop app as a thin client (no local models, no local index):

    RAG_BACKEND_URL=http://localhost:8800 python app.py

In this mode Chat, Dashboard, Reports and System all read from the backend;
the ingestion buttons are disabled (index the usual way:
`docker compose run --rm app index /files`). Answers are produced by the
same Agent code path as local mode, so accuracy is identical.

Endpoints: GET /health /status /charts /reports · POST /ask /report
/clear_history (JSON).


## Hardening the backend (pilot -> production)

    # docker-compose.yml, api service:
    environment:
      RAG_API_KEY: "<shared secret>"     # clients send X-Api-Key
      RAG_WATCH_DIR: /files              # auto-ingest data drops (default)
      RAG_WATCH_INTERVAL: "60"

Web UI: open http://<host>:8800/ in a browser (chat + reports + status;
enter the API key once, it is remembered per browser). TLS: put a reverse
proxy in front, e.g. Caddy:

    reverse_proxy https://assistant.kronospan.local -> localhost:8800

Nightly verification (crontab on the host) keeps the System-page scoreboard
fresh:

    0 6 * * * cd /path/to/project && ./venv/bin/python -m eval.run_use_cases --data /path/to/files >> /var/log/jarvis-checks.log 2>&1
    15 6 * * * cd /path/to/project && ./venv/bin/python test_queries.py >> /var/log/jarvis-checks.log 2>&1

Audit trail: every question/answer (app, API and web) is appended to
<RAG_DATA_DIR>/audit/audit-YYYYMM.jsonl — see SECURITY.md.
