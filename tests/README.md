# tests — deterministic regression suite

Offline, no Ollama, no Excel files. Runs in ~1.5s. Covers the pure-Python,
deterministic layers (the parts that fail SILENTLY and are mockable), plus the
sandbox security boundary.

    pip install pytest
    pytest                       # from the repo root

## What is locked here

- **test_sandbox.py** — the AST gate rejects imports / dunder escapes / file &
  OS access; legit analysis code passes; the child `RLIMIT_AS` cap contains a
  memory bomb instead of OOM-ing the host.
- **test_value_index.py** — name resolution: the documented one- and two-typo
  cases resolve, nonsense never auto-resolves, and `_FUZZY_TOKEN_RATIO <= 0.667`
  (regression guard for the old 0.78/0.6 comment drift that would break the
  two-typo case).
- **test_ingestion.py** — the "red-team finds": dotted CODES are not converted
  as EU money; a merged parent is not forward-filled across a gap; a leading
  title row is dropped; duplicate columns are de-duped and quotes stripped.
- **test_semantic_model.py** — column meaning classification (number / date /
  text / empty / year) and Total/Subtotal row detection.
- **test_query_plan.py** — `compile_plan` excludes Total rows, lists distinct
  rows, REJECTS unknown columns/aggregations (no silent guessing), and emits
  every literal via `repr()` so a value that looks like code is inert.

## Adding a regression

When a model picks a wrong column or a sheet shape breaks ingestion, reproduce
it as a 3-5 row DataFrame fixture and assert the fixed behaviour here. Each
"red-team find" comment in the source should have a matching test.
