# Architecture

The package is layered by concern. Dependencies flow in ONE direction —
upper layers import lower ones, never the reverse — so each layer can be read,
tested, and changed in isolation.

    ui            gui, workers              PyQt6 desktop; talks ONLY to agent
     │
    agent         agent                     routing + dispatch; the orchestrator
     │
    planning      planner, query_plan,      question -> validated plan -> pandas
     │            reasoner, tier05,
     │            followup, proposals
     │
    semantics     semantic_model,           what the data MEANS: column profiles,
     │            value_index, grounding,   value index, grounding, ambiguity
     │            ambiguity, table_cards
     │
    ingest        ingestion, column_types   Excel/PDF -> clean DataFrames
    retrieval     rag, retrieval,           PDF prose RAG (parallel to ingest)
                  vector_store, embedding_cache
     │
    runtime       sandbox, numfmt,          execution boundary + output shaping
     │            house_rules
    llm           ollama_client, llm_json   model I/O
     │
    config        config                    leaf: imported by everyone, imports
                                            nothing internal

Key rules that keep the layering honest:

- **`config` is a leaf.** It imports only the stdlib. Everything imports it;
  it imports nothing of ours. (If you ever need config to import a sibling,
  that sibling belongs above config, not config below it.)
- **`runtime.sandbox` is the only place model-written code executes.** Both the
  deterministic plan compiler (`planning.query_plan`) and the codegen fallback
  (`agent`) funnel through it, so the AST gate + resource caps apply once,
  everywhere.
- **`agent` is the only thing `ui` imports.** The GUI never reaches into
  planning/semantics directly — swap the UI (CLI, web) without touching logic.
- **`ingest` and `retrieval` are siblings**, not a chain: tabular questions go
  ingest -> semantics -> planning; prose questions go retrieval. `agent` routes
  between them.

`spark_eval.py` (live accuracy harness) and `app.py` (entry) sit at the repo
root because they are scripts, not library code; both import the package.
