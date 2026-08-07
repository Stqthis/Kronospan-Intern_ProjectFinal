import os

# --------------------------------------------------------------------------- #
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Remote backend: when set (e.g. http://dgx:8800) the desktop app is a thin
# client of the Dockerised API (jarvisman.server) — the model and index run
# on the server; nothing is loaded locally.
BACKEND_URL = os.environ.get("RAG_BACKEND_URL", "").strip()


# Target machine is the DGX Spark: the 32b is the DEFAULT for every call
# (plan, repair, codegen, synthesis, index-time cards). A CPU laptop
# overrides DOWN via env: RAG_CHAT_MODEL=qwen2.5-coder:14b.
DEFAULT_CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "qwen2.5-coder:32b")
# NOTE: changing the embed model invalidates persisted PDF vectors --
# re-index PDFs after switching machines/models. Excel reasoning does not
# use embeddings, so tables are unaffected.
DEFAULT_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
# Planning is a small structured output and latency-critical; the Spark's
# memory bandwidth makes the 32b ~2x slower per token, so the planner can run
# a smaller model while synthesis/cards keep the big one. Empty = chat model.
PLAN_MODEL = os.environ.get("RAG_PLAN_MODEL", "")
# --------------------------------------------------------------------------- #
# Role-based model routing                                                    #
# --------------------------------------------------------------------------- #
CODE_MODEL = os.environ.get("RAG_CODE_MODEL", "")
UNDERSTANDING_MODEL = os.environ.get("RAG_UNDERSTANDING_MODEL", "")


def model_for(role: str, chat_model: str) -> str:
    """Resolve a task ROLE to a concrete model name, falling back to the
    given chat_model when the role-specific model is not configured.

    Also records the role so the timing module can attribute the model call
    that follows. Every call site evaluates this inline as chat()'s first
    argument, so the role is always the one about to run.
    """
    try:
        from jarvisman.runtime import timing
        timing.set_pending(role)
    except Exception:
        pass
    if role == "plan":
        return CODE_MODEL or PLAN_MODEL or chat_model
    if role == "codegen":
        return CODE_MODEL or chat_model
    if role in ("route", "pick_table", "propose", "synthesize", "chat",
                "understand"):
        return UNDERSTANDING_MODEL or chat_model
    return chat_model


CHAT_TIMEOUT = (10, 600)
EMBED_TIMEOUT = (10, 180)
LIST_TIMEOUT = (5, 20)


# "-1" keeps the model pinned in memory (never unload) so answers never pay a
# reload. There is ample unified memory to hold it; set a duration like "30m"
# to release it when idle.
KEEP_ALIVE = os.environ.get("RAG_KEEP_ALIVE", "-1")
EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH", "64"))
EMBED_CONCURRENCY = int(os.environ.get("RAG_EMBED_CONCURRENCY", "4"))

# --------------------------------------------------------------------------- #
# Retrieval / chunking                                                        #
# --------------------------------------------------------------------------- #
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RAG_TOP_K = 5

# Table profiling — accuracy mode. Profiling builds the per-column meaning map
# the reasoner uses to bind questions to the right column. Do NOT disable.
SKIP_TABLE_PROFILING = False
TABLE_CARDS = True


FETCH_K = 20               # candidates pulled from EACH retriever before fusion
TOP_K = 5                  # chunks finally fed to the model
MAX_CONTEXT_CHARS = 6000   # hard cap on context handed to the model

# Hybrid retrieval: dense (FAISS) + lexical (BM25) fused with Reciprocal Rank
# Fusion.
RRF_K = 60
HYBRID_DENSE_WEIGHT = 0.8
HYBRID_SPARSE_WEIGHT = 0.2
BM25_K1 = 1.5
BM25_B = 0.75

# --------------------------------------------------------------------------- #
# Sandbox (plot code execution)                                               #
# --------------------------------------------------------------------------- #
SANDBOX_TIMEOUT = 20
PLOT_DPI = 100
PLOT_MAX_WIDTH = 560
SANDBOX_MEM_LIMIT_MB = int(os.environ.get("RAG_SANDBOX_MEM_MB", "4096"))
SANDBOX_CPU_LIMIT_S = int(
    os.environ.get("RAG_SANDBOX_CPU_S", str(SANDBOX_TIMEOUT + 10)))

# --------------------------------------------------------------------------- #
# Data query / analysis tool                                                  #
# --------------------------------------------------------------------------- #
QUERY_SAMPLE_ROWS = 10
QUERY_MAX_RESULT_ROWS = 100
# The plain-text rendering is read by the LLM (synthesis) and the eval, so it
# stays small. The HTML goes to the UI, which has a scrollable TableWindow and
# can hold far more. Two consumers, two caps.
QUERY_MAX_TABLE_ROWS = int(os.environ.get("RAG_MAX_TABLE_ROWS", "2000"))
CATEGORICAL_MAX_UNIQUE = 200
ANALYSIS_MAX_RETRIES = 2

# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #
DATA_DIR = os.environ.get(
    "RAG_DATA_DIR", os.path.join(os.path.expanduser("~"), ".offline_rag_assistant")
)
INDEX_DIR = os.path.join(DATA_DIR, "index")
# Persist solved query plans across restarts: a repeated/reworded question is
# then answered with zero model calls. Keyed by (index version, question).
PLAN_CACHE_PERSIST = os.environ.get("RAG_PLAN_CACHE_PERSIST", "1") == "1"
PLAN_CACHE_PATH = os.path.join(INDEX_DIR, "plan_cache.json")
EMBED_CACHE_PATH = os.path.join(DATA_DIR, "embed_cache.pkl")

SUPPORTED_EXTENSIONS = (".pdf", ".xlsx", ".xls")

# --------------------------------------------------------------------------- #
# Excel ingestion (speed + accuracy)                                          #
# --------------------------------------------------------------------------- #
# Hard row cap per sheet. The old value (5000) silently truncated the daily
# loan schedules (LTL ~78k rows, CY28 ~109k), which made "as at <date>" balances
# wrong. Keep a high ceiling so real data is complete; lower it only on tiny
# machines. Set 0 for "no limit".
_excel_rows = int(os.environ.get("RAG_EXCEL_MAX_ROWS", "250000"))
EXCEL_MAX_ROWS = None if _excel_rows <= 0 else _excel_rows

# Sheets skipped at ingest time (case-insensitive). Derived pivots, per-country
# breakdowns, CEO views, and password/helper tabs are not primary data and only
# slow indexing and confuse table selection. Real data sheets (MATRIX,
# CY05-Query, WCR MATRIX, LTL_Data, BIG DATA) are kept. All three lists are
# comma-separated env-overridable.
SKIP_SHEET_NAMES = [s.strip().lower() for s in os.environ.get(
    "RAG_SKIP_SHEETS",
    "password,index,query new,3mth exp,group-ceo,pivot input .v2").split(",") if s.strip()]
SKIP_SHEET_PREFIXES = [s.strip().lower() for s in os.environ.get(
    "RAG_SKIP_SHEET_PREFIXES", "p-").split(",") if s.strip()]
SKIP_SHEET_SUBSTRINGS = [s.strip().lower() for s in os.environ.get(
    "RAG_SKIP_SHEET_SUBSTR", "ceo,pivot").split(",") if s.strip()]
# Content-based junk filter: a recovered table this wide with this few distinct
# column base-names is a pivot dump, not data.
DEGENERATE_MIN_WIDTH = int(os.environ.get("RAG_DEGENERATE_MIN_WIDTH", "8"))
DEGENERATE_MAX_DISTINCT = int(os.environ.get("RAG_DEGENERATE_MAX_DISTINCT", "3"))

# --------------------------------------------------------------------------- #
# LLM context window                                                          #
# --------------------------------------------------------------------------- #
NUM_CTX = int(os.environ.get("RAG_NUM_CTX", "8192"))

# --------------------------------------------------------------------------- #
# Determinism / reproducibility                                               #
# --------------------------------------------------------------------------- #
# The single biggest lever for answer CONSISTENCY. A fixed seed makes Ollama's
# sampling reproducible, so the same question yields the same plan, the same
# generated code, and the same phrased answer on every run instead of a fresh
# dice roll. Combined with greedy decoding (temperature 0) on the fact-
# producing stages, identical input -> identical output. Set RAG_SEED to any
# int to shift the (repeatable) tie-breaks; RAG_DETERMINISTIC=0 restores the
# old sampled behaviour if you ever want variety over reproducibility.
DETERMINISTIC = os.environ.get("RAG_DETERMINISTIC", "1") == "1"
LLM_SEED = int(os.environ.get("RAG_SEED", "42"))

# --------------------------------------------------------------------------- #
# Semantic model / reasoning layer                                            #
# --------------------------------------------------------------------------- #
PROFILE_TOP_K = 12
PROMPT_DIM_MAX_VALUES = 10
VALUE_INDEX_MAX_CARDINALITY = int(
    os.environ.get("RAG_VALUE_MAX_CARD", "50000"))
RELATIONSHIP_MIN_COVERAGE = 0.95
RELATIONSHIP_MAX_DISTINCT = 50000
PLAN_TEMPERATURE = 0.0
# A query plan is a small JSON object; cap generation so the model can't run
# on past it. Raise via env if plans for very wide schemas get truncated.
PLAN_NUM_PREDICT = int(os.environ.get("RAG_PLAN_NUM_PREDICT", "768"))
PLAN_REPAIR_ATTEMPTS = 1
# Phrase compact (scalar / few-row) results as a direct sentence instead of a
# raw value, so "any changes?" gets a Yes/No + reason rather than a data dump.
# Only fires for results <= SYNTHESIZE_MAX_ROWS and has a number-echo guard, so
# figures are never invented. Set RAG_SYNTH=0 for maximum speed (raw results).
SYNTHESIZE_ANSWER = os.environ.get("RAG_SYNTH", "1") == "1"
# A phrased answer is 1-3 sentences; cap generation so decode stops early.
SYNTH_NUM_PREDICT = int(os.environ.get("RAG_SYNTH_NUM_PREDICT", "220"))

# --------------------------------------------------------------------------- #
# UI animations                                                               #
# --------------------------------------------------------------------------- #
# Motion: sidebar slide, theme crossfade, row-by-row table reveal. Set
# RAG_ANIMATIONS=0 to disable all motion (accessibility / low-power screens).
ANIMATIONS = os.environ.get("RAG_ANIMATIONS", "1") == "1"
ANIM_DURATION_MS = int(os.environ.get("RAG_ANIM_MS", "240"))
ANIM_TABLE_INTERVAL_MS = int(os.environ.get("RAG_ANIM_ROW_MS", "45"))
ANIM_TABLE_MAX_ROWS = int(os.environ.get("RAG_ANIM_MAX_ROWS", "60"))
# Chart entrance animation (bars grow, pie sweeps, lines draw in).
CHART_ANIM_FRAMES = int(os.environ.get("RAG_CHART_ANIM_FRAMES", "26"))
CHART_ANIM_INTERVAL_MS = int(os.environ.get("RAG_CHART_ANIM_MS", "22"))
SYNTHESIZE_MAX_ROWS = int(os.environ.get("RAG_SYNTH_MAX_ROWS", "3"))
# Comparison/opinion questions need prose even when the result has many rows.
# Pure figure questions do not: the table speaks for itself and the extra LLM
# call is pure latency. Capped so a huge table is never fed to the model.
SYNTHESIZE_PROSE_MAX_ROWS = int(os.environ.get("RAG_SYNTH_PROSE_MAX_ROWS", "40"))
SYNTH_PROSE_NUM_PREDICT = int(os.environ.get("RAG_SYNTH_PROSE_NUM_PREDICT", "420"))
# Results bigger than this open in TableWindow behind a chip instead of being
# rendered inline (QTextBrowser lays tables out to viewport width).
UI_TABLE_INLINE_MAX_ROWS = int(os.environ.get("RAG_UI_TABLE_ROWS", "30"))
UI_TABLE_INLINE_MAX_COLS = int(os.environ.get("RAG_UI_TABLE_COLS", "10"))
MAX_ANALYSIS_TABLES_REASONER = int(os.environ.get("RAG_MAX_TABLES", "6"))
PROMPT_MEANING_MAX_CHARS = 80

# --------------------------------------------------------------------------- #
# Feature flags (each phase independently switchable; 1/0 via env)            #
# --------------------------------------------------------------------------- #
FOLLOWUP_ENABLED = os.environ.get("RAG_FOLLOWUP", "1") == "1"
CLARIFY_ENABLED = os.environ.get("RAG_CLARIFY", "1") == "1"
CHART_PLAN_ENABLED = os.environ.get("RAG_CHART_PLAN", "1") == "1"
VIRTUAL_LONG_VIEWS = os.environ.get("RAG_LONG_VIEWS", "1") == "1"
EU_NUMBER_PARSE = os.environ.get("RAG_EU_NUMBERS", "1") == "1"
MULTI_TABLE_SHEETS = os.environ.get("RAG_MULTI_TABLE", "0") == "1"
UI_LANG = os.environ.get("RAG_UI_LANG", "en")

# UI theme: Dark | Light | Emerald | Purple (see ui/theme_manager.py)
UI_THEME = os.environ.get("RAG_UI_THEME", "Dark")
DISTINCT_SELECT = os.environ.get("RAG_DISTINCT", "1") == "1"
COMPANION_CODES = os.environ.get("RAG_CODES", "1") == "1"
NUMBER_FORMAT = os.environ.get("RAG_NUM_FORMAT", "locale")

# --------------------------------------------------------------------------- #
# Assistant persona (cosmetic) + interactive charts                          #
# --------------------------------------------------------------------------- #
ASSISTANT_NAME = os.environ.get("RAG_ASSISTANT_NAME", "Jarvis").strip() or "Jarvis"
USER_HONORIFIC = os.environ.get("RAG_USER_HONORIFIC", "sir")
CHART_TOP_N = int(os.environ.get("RAG_CHART_TOP_N", "20"))
CHART_VALUE_LABELS = os.environ.get("RAG_CHART_LABELS", "1") == "1"

# ---- Spark-era features (each independently switchable) -------------------- #
# NOTE: TABLE_CARDS and SKIP_TABLE_PROFILING are defined ONCE in the
# retrieval/chunking section above. Do not redefine them here.
SCHEMA_SAMPLES = os.environ.get("RAG_SCHEMA_SAMPLES", "1") == "1"
TIER05_ENABLED = os.environ.get("RAG_TIER05", "1") == "1"
PLAN_CACHE = os.environ.get("RAG_PLAN_CACHE", "1") == "1"
CONDITION_COVERAGE = os.environ.get("RAG_COVERAGE", "1") == "1"
PLAN_V2 = os.environ.get("RAG_PLAN_V2", "1") == "1"
EXPLAIN = os.environ.get("RAG_EXPLAIN", "0") == "1"
INTERPRET_CLARIFY = os.environ.get("RAG_INTERPRET_CLARIFY", "1") == "1"
COLUMN_CLARIFY = os.environ.get("RAG_COLUMN_CLARIFY", "0") == "1"
PROPOSE_ALTERNATIVES = os.environ.get("RAG_PROPOSE", "1") == "1"
HOUSE_RULES_PATH = os.environ.get(
    "RAG_HOUSE_RULES", os.path.join(DATA_DIR, "rules.txt"))
INTERPRET_DECIDED_WORDS = [
    w.strip() for w in os.environ.get(
        "RAG_DECIDED_WORDS",
        "equivalent,equiv,converted,conversion,denominated,"
        "ισοδυναμ,μετατρ,εκφρασμεν").split(",") if w.strip()]
INTERPRET_CURRENCY_WORDS = [
    w.strip() for w in os.environ.get(
        "RAG_CURRENCY_WORDS", "currency,ccy,νομισμα").split(",") if w.strip()]
PLAN_ZERO_IS_ANSWER = os.environ.get("RAG_PLAN_ZERO_IS_ANSWER", "1") == "1"


# --------------------------------------------------------------------------- #
# Deterministic-path diagnostics                                              #
# --------------------------------------------------------------------------- #
def dbg(where: str, exc: BaseException) -> None:
    """Surface a swallowed exception from the DETERMINISTIC code paths."""
    if not EXPLAIN:
        return
    try:
        import sys
        print(f"[deterministic-degrade] {where}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
    except Exception:
        pass