# Improvement roadmap — Kronospan financial data assistant

Prioritised recommendations based on verifying the pipeline against the real
use-case data (July 2026). Ordered by impact on answer accuracy for treasury
users.

## 1. Data feeds (biggest accuracy lever — no code required)
- **Monthly DATA snapshots, not PDF reports.** Use cases 5, 6 and 8
  (directorship changes, Mexico directors, South-group name changes) need TWO
  dated CY05 `*-DATA.xlsx` snapshots to compare. The index currently holds
  only Sep-2023; Mar'24/Sep'24 exist only as PDFs, which feed text search,
  not table comparison. A monthly drop of CY01/CY05/WCR/LTL DATA exports
  makes the whole "changes between dates" use-case family answerable.
- **Fix the swapped WCR files at source** (`WCR_30_07_2024.xlsx` holds the
  30/12/2024 report and vice versa) and validate filename-vs-content dates at
  ingestion time — the app routes "as at <date>" questions by these.
- **Standardise entity spellings** in exports ("Kronospan CR, spol s r.o."
  vs "spol. s r.o.") or add an alias list; exact-name filters silently miss.
- Refresh the LTL export on the index (69k-row version predates the current
  77.7k-row file with rates through 2025-06).

## 2. Accuracy infrastructure
- **Golden-set CI**: `eval/run_use_cases.py` (13 deterministic checks) after
  every ingestion, `test_queries.py` (18 graded LLM questions) nightly on the
  Spark; alert on regressions. Extend the question set as users report gaps.
- **Table extraction from PDF reports** (CY05 monthly PDFs are structured):
  parse them into snapshot tables so history exists even where only PDFs
  survive.
- **Update the use-case document**: 11.1.1 has a typo (31,499,912.50) and
  11.2's rates (1.000/1.901/2.901) describe the 2021-23 period, not current
  data — testers grading against it will file false bugs.

## 3. Models (DGX Spark)
- Split roles: keep `qwen2.5-coder:32b` for plan/codegen, set
  `RAG_UNDERSTANDING_MODEL=llama3:70b` for routing/synthesis — better prose,
  and both stay resident (`OLLAMA_MAX_LOADED_MODELS=2`, already configured).
- Consider a newer coder model when revalidating (rerun the golden set to
  compare before switching); keep temperature 0 for plans/codegen.
- Embeddings: `nomic-embed-text` is fine for 19 PDFs; revisit only if the
  document base grows into the hundreds.

## 4. Backend service (now in place)
- `jarvisman.server` in Docker is the single source of answers
  (`docker compose up -d ollama api`); desktop clients connect with
  `RAG_BACKEND_URL`. Next steps: an API key header, HTTPS behind a reverse
  proxy, and per-question audit logging (question, generated code, tables
  used, answer) — finance users will eventually need an audit trail.
- Scheduled ingestion: a watch-folder or cron in the api container
  (`docker compose run --rm app index /files`) turns the monthly data drop
  into a hands-off refresh.

## 5. Product polish
- Snapshot-compare UI: pick two dates, get the diff table (directors, names,
  balances) computed deterministically — no LLM in the loop.
- Saved/favourite questions per user; one-click export of any answer table
  to Excel (the Reports page already exports; extend to chat answers).
- Greek UI already exists (`RAG_UI_LANG=el`); translate the new pages if
  Greek-speaking treasury staff will use them.
