# Offline Servant — local RAG assistant (production)

The runtime code lives in the `jarvisman/` package, organised by concern (see
**Project layout** below). `app.py`/`spark_eval.py` are thin entry scripts at
the repo root; `tests/` holds an offline deterministic regression suite
(`pytest`).

## Requirements
- Python 3.10+ ; running Ollama (`ollama serve`)
- ollama pull qwen2.5-coder:32b   (code model — plans & queries)
- ollama pull llama3:70b          (understanding model — descriptions, reasoning, clarifications, proposals)
- ollama pull nomic-embed-text    (PDF embedding search)
- pip install -r requirements.txt

## Run
    python app.py                  # desktop GUI (recommended entry)
    python -m jarvisman.ui.gui     # equivalent
    pytest                         # run the offline test suite

## Project layout
    jarvisman/                 the package (import as `jarvisman.<area>.<module>`)
      config.py               every module imports this; env-driven knobs
      agent.py                top-level orchestrator (routing + dispatch)
      llm/                    ollama_client, llm_json        (model I/O)
      ingest/                 ingestion, column_types        (Excel/PDF -> tables)
      semantics/              semantic_model, value_index, grounding,
                              ambiguity, table_cards         (what the data MEANS)
      planning/               planner, query_plan, reasoner, tier05,
                              followup, proposals            (question -> plan)
      retrieval/              rag, retrieval, vector_store,
                              embedding_cache                (PDF prose RAG)
      runtime/                sandbox, numfmt, house_rules   (execution + output)
      ui/                     gui, workers, charting, chart_window
                              (PyQt6 desktop + interactive matplotlib charts)
    app.py                    entry point (calls jarvisman.ui.gui:main)
    spark_eval.py             live accuracy harness (see below)
    tests/                    offline deterministic regression suite

    # PyInstaller: collect the whole package, e.g.
    #   pyinstaller --collect-submodules jarvisman app.py

## Persona & interactive charts
The assistant has a light cosmetic persona shown in the greeting and chat
header (no effect on logic):

    RAG_ASSISTANT_NAME=Jarvis     # name in the greeting / chat header
    RAG_USER_HONORIFIC=sir        # how it addresses you; set to 'master' for
                                  # "How may I help you, master?" — or "" for none

Any answer that returns a table with a numeric column shows a **Visualize /
breakdown** link. It opens an interactive chart window (zoom / pan / save via
the matplotlib toolbar, hover tooltips) where you can switch chart type, pick
the axis and measure, add a breakdown (second category -> grouped bars), and
cap the top-N. Charts use the existing matplotlib dependency — no web view.

    RAG_CHART_TOP_N=20            # default category cap before "Other"
    RAG_CHART_LABELS=1            # value labels on bars

## Two-model routing (set on the Spark)
    export RAG_CODE_MODEL=qwen2.5-coder:32b
    export RAG_UNDERSTANDING_MODEL=llama3:70b
    export OLLAMA_KEEP_ALIVE=30m
The understanding model is used for: table/column DESCRIPTIONS at index time,
request routing, table selection, clarifications, proposals, and answer
phrasing. The code model writes the plan and pandas/matplotlib code. Leave the
vars unset to run everything on one model.

## How it scales to many tables (no hardcoding)
At index time the understanding model READS each table's columns + sample rows
and writes, per table: a one-line summary of what each row represents, and per
column a business-language meaning that flags the CANONICAL figure when several
columns look alike (e.g. "AMOUNT EURO is the primary converted EUR value;
Finance/ECCM is an internal label, not money"). These descriptions are cached
to table_profile.json and injected into the planner, so the model picks the
right column instead of guessing — for ANY table, generated automatically.
Rebuild the index once to (re)generate descriptions.

## rules.txt — the override layer (optional)
For the few tables where an auto-description is wrong, OR for standing
policies, add plain-language lines to rules.txt (or set RAG_HOUSE_RULES).
They are injected into the planner and code generator on every question.
This is the OVERRIDE for the automatic descriptions, not the primary
mechanism — you do not write rules for every table, only exceptions.

## First run
Add PDF/Excel files, click Build / Update Index once (this runs the
description pass — slower with llama3:70b, but cached). Rebuild after
replacing data files.

## Debugging on the Spark
RAG_EXPLAIN=1 prints the decision trace (which table, which columns, which
model per step) — use it to see why a column was chosen or a question asked.

## Measuring REAL accuracy on the Spark (spark_eval.py)
The only honest accuracy number comes from running real questions through the
real models on your data. After building the index in the app:

    RAG_CODE_MODEL=qwen3:30b RAG_UNDERSTANDING_MODEL=llama3:70b \
    RAG_EXPLAIN=1 python spark_eval.py questions.jsonl

Edit questions.jsonl (one {"q":..., "expect":...} per line; 'expect' is a
string or list of strings that must appear in a correct answer). Write the
questions the way a non-expert user would. It prints per-question pass/fail,
the table/columns the model chose, latency, and writes spark_eval_results.json.
