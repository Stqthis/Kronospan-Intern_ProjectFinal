import os

# --------------------------------------------------------------------------- #
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


# Target machine is the DGX Spark: the 32b is the DEFAULT for every call
# (plan, repair, codegen, synthesis, index-time cards). A CPU laptop
# overrides DOWN via env: RAG_CHAT_MODEL=qwen2.5-coder:14b.
DEFAULT_CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "qwen2.5-coder:32b")
# NOTE: changing the embed model invalidates persisted PDF vectors --
# re-index PDFs after switching machines/models. Excel reasoning does not
# use embeddings, so tables are unaffected.
# The default is a REAL embedding model (the README and the in-app
# troubleshooting both require one); a chat model silently produces poor
# vectors. Deployments that deliberately embed with the chat model (e.g. the
# Spark) set RAG_EMBED_MODEL explicitly and re-index.
DEFAULT_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
# Planning is a small structured output and latency-critical; the Spark's
# memory bandwidth makes the 32b ~2x slower per token, so the planner can run
# a smaller model while synthesis/cards keep the big one. Empty = chat model.
PLAN_MODEL = os.environ.get("RAG_PLAN_MODEL", "")
# --------------------------------------------------------------------------- #
# Role-based model routing                                                    #
# --------------------------------------------------------------------------- #
# Two jobs, two strengths. A CODE model (qwen2.5-coder) is best at producing
# the JSON plan and pandas/matplotlib code; a REASONING/language model
# (e.g. llama3:70b) is better at UNDERSTANDING -- classifying intent, picking
# the table, reading the data's meaning, proposing alternatives, and phrasing
# the final answer. Either may be empty, in which case that role falls back
# to the general chat model, so a single-model setup keeps working unchanged.
#
#   CODE_MODEL          -> plan + codegen (defaults to PLAN_MODEL or chat)
#   UNDERSTANDING_MODEL -> routing, table choice, proposals, synthesis
#
# Resolution is centralised in model_for(role); call sites pass a role name,
# never a raw model string.
CODE_MODEL = os.environ.get("RAG_CODE_MODEL", "")
UNDERSTANDING_MODEL = os.environ.get("RAG_UNDERSTANDING_MODEL", "")


def model_for(role: str, chat_model: str) -> str:
    """Resolve a task ROLE to a concrete model name, falling back to the
    given chat_model when the role-specific model is not configured.

    Roles:
      'plan'        -> CODE_MODEL or PLAN_MODEL or chat_model
      'codegen'     -> CODE_MODEL or chat_model
      'route'       -> UNDERSTANDING_MODEL or chat_model
      'pick_table'  -> UNDERSTANDING_MODEL or chat_model
      'propose'     -> UNDERSTANDING_MODEL or chat_model
      'synthesize'  -> UNDERSTANDING_MODEL or chat_model
      'chat'        -> UNDERSTANDING_MODEL or chat_model
    Unknown roles fall back to chat_model.
    """
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


KEEP_ALIVE = os.environ.get("RAG_KEEP_ALIVE", "30m")
EMBED_BATCH_SIZE = 16  

# --------------------------------------------------------------------------- #
# Retrieval / chunking                                                        #
# --------------------------------------------------------------------------- #
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 150        # characters of overlap between consecutive chunks
RAG_TOP_K = 3              # Retrieve only top 3 most relevant chunks
FETCH_K = 20               # candidates pulled from EACH retriever before fusion
TOP_K = 5                  # chunks finally fed to the model
MAX_CONTEXT_CHARS = 6000   # hard cap on context handed to the model

# Hybrid retrieval: dense (FAISS) + lexical (BM25) fused with Reciprocal Rank
# Fusion. RRF needs no score normalisation; the weights bias the blend.
RRF_K = 60                 # RRF damping constant (standard value from the paper)
HYBRID_DENSE_WEIGHT = 0.8  # semantic weight in the fusion
HYBRID_SPARSE_WEIGHT = 0.2 # lexical (BM25) weight in the fusion
BM25_K1 = 1.5
BM25_B = 0.75

# --------------------------------------------------------------------------- #
# Sandbox (plot code execution)                                               #
# --------------------------------------------------------------------------- #
SANDBOX_TIMEOUT = 20       # seconds before a runaway plot/query job is killed
PLOT_DPI = 100
PLOT_MAX_WIDTH = 560       # px, display width inside the chat view
# Child-process resource caps (POSIX only; silently skipped where the
# `resource` module is unavailable, e.g. Windows / some frozen builds). The
# wall-clock SANDBOX_TIMEOUT terminates a hung child, but a runaway ALLOCATION
# (a generated cross-join, a 10**6 x 10**6 frame) can OOM the host before that
# timer fires. RLIMIT_AS caps the child's address space; RLIMIT_CPU is a
# backstop for a tight C-loop that never yields to the timeout. 0 disables a
# given cap.
#   NOTE: RLIMIT_AS limits VIRTUAL address space, which counts mmap'd-but-
#   untouched arenas reserved by NumPy/BLAS. Keep it GENEROUS (default 4 GB) so
#   legitimate pandas work never trips it -- the goal is to stop gross bombs,
#   not to micro-budget memory. The cap is applied AFTER the heavy imports, so
#   importing pandas/numpy/matplotlib can never fail on it.
SANDBOX_MEM_LIMIT_MB = int(os.environ.get("RAG_SANDBOX_MEM_MB", "4096"))
SANDBOX_CPU_LIMIT_S = int(
    os.environ.get("RAG_SANDBOX_CPU_S", str(SANDBOX_TIMEOUT + 10)))

# --------------------------------------------------------------------------- #
# Data query / analysis tool                                                  #
# --------------------------------------------------------------------------- #
QUERY_SAMPLE_ROWS = 10      # rows of each sheet shown to the model for grounding
QUERY_MAX_RESULT_ROWS = 100  # cap on rows displayed from a tabular result
CATEGORICAL_MAX_UNIQUE = 200  # list a column's distinct values if it has <= this many
ANALYSIS_MAX_RETRIES = 2     # self-debug attempts after the first code generation

# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #
DATA_DIR = os.environ.get(
    "RAG_DATA_DIR", os.path.join(os.path.expanduser("~"), ".offline_rag_assistant")
)
INDEX_DIR = os.path.join(DATA_DIR, "index")
EMBED_CACHE_PATH = os.path.join(DATA_DIR, "embed_cache.pkl")

SUPPORTED_EXTENSIONS = (".pdf", ".xlsx", ".xls")
# --------------------------------------------------------------------------- #
# LLM context window                                                          #
# --------------------------------------------------------------------------- #
# Ollama's DEFAULT num_ctx is small (2048-4096 tokens) and it silently
# truncates longer prompts from the front -- the model then never sees the
# schema or the rules. Every chat call now sends this unless the caller
# overrides it.
#
# Keep it as SMALL as the prompts allow: on CPU, llama.cpp pre-allocates the
# KV cache for the whole window, so 16k on a 14B model costs gigabytes of RAM
# (and swap-death on a laptop). 8192 fits the planner/codegen prompts with
# room to spare. CRITICAL: use ONE value everywhere -- Ollama RELOADS the
# model whenever num_ctx changes between calls (column_types also uses 8192).
NUM_CTX = int(os.environ.get("RAG_NUM_CTX", "8192"))

# --------------------------------------------------------------------------- #
# Semantic model / reasoning layer                                            #
# --------------------------------------------------------------------------- #
PROFILE_TOP_K = 12                # top values stored per column in the profile
PROMPT_DIM_MAX_VALUES = 10        # dimension values shown in the planner prompt
# Index a text column up to this many distinct values. Registry-style data
# (company names, directors) easily exceeds 5000; a column ABOVE the cap is
# invisible to grounding, so its questions silently fall to the codegen tier
# where text matching is model-dependent. The index is pure Python and built
# once per (re)index, so a high cap costs memory, not latency.
VALUE_INDEX_MAX_CARDINALITY = int(
    os.environ.get("RAG_VALUE_MAX_CARD", "50000"))
RELATIONSHIP_MIN_COVERAGE = 0.95  # child-in-parent containment for an FK link
RELATIONSHIP_MAX_DISTINCT = 50000 # skip relationship check above this size
PLAN_TEMPERATURE = 0.0
PLAN_REPAIR_ATTEMPTS = 1          # validation-driven repair calls
# Phrasing the result costs a SECOND full LLM round-trip per question; on CPU
# that roughly doubles latency. Default off: the reasoner formats the result
# + provenance deterministically. Set RAG_SYNTH=1 for LLM-phrased answers.
SYNTHESIZE_ANSWER = os.environ.get("RAG_SYNTH", "0") == "1"
# When synthesis is on, only phrase results with at most this many rows
# (a scalar/few-row answer benefits from a sentence; a big table does
# not and the LLM call is pure latency). 0 disables the row cap.
SYNTHESIZE_MAX_ROWS = int(os.environ.get("RAG_SYNTH_MAX_ROWS", "3"))
MAX_ANALYSIS_TABLES_REASONER = int(os.environ.get("RAG_MAX_TABLES", "6"))  # tables exposed to the planner, best first
PROMPT_MEANING_MAX_CHARS = 80     # cap per-column meaning text in prompts

# --------------------------------------------------------------------------- #
# Feature flags (each phase independently switchable; 1/0 via env)            #
# --------------------------------------------------------------------------- #
FOLLOWUP_ENABLED = os.environ.get("RAG_FOLLOWUP", "1") == "1"      # Phase A
CLARIFY_ENABLED = os.environ.get("RAG_CLARIFY", "1") == "1"        # Phase B: value-level "did you mean" only
CHART_PLAN_ENABLED = os.environ.get("RAG_CHART_PLAN", "1") == "1"  # Phase C
VIRTUAL_LONG_VIEWS = os.environ.get("RAG_LONG_VIEWS", "1") == "1"  # Phase D
EU_NUMBER_PARSE = os.environ.get("RAG_EU_NUMBERS", "1") == "1"     # Phase D
MULTI_TABLE_SHEETS = os.environ.get("RAG_MULTI_TABLE", "0") == "1" # Phase D (opt-in: default one sheet = one table)
UI_LANG = os.environ.get("RAG_UI_LANG", "en")                      # Phase E: en | el
DISTINCT_SELECT = os.environ.get("RAG_DISTINCT", "1") == "1"       # dedup row listings
COMPANION_CODES = os.environ.get("RAG_CODES", "1") == "1"          # NAME column brings its CODE column
NUMBER_FORMAT = os.environ.get("RAG_NUM_FORMAT", "locale")         # locale | us | eu | plain
# --------------------------------------------------------------------------- #
# Assistant persona (cosmetic) + interactive charts                          #
# --------------------------------------------------------------------------- #
# The name shown in the chat header and welcome greeting. Purely presentational
# -- it changes no logic. Override with RAG_ASSISTANT_NAME; an empty string
# falls back to a neutral "Assistant".
ASSISTANT_NAME = os.environ.get("RAG_ASSISTANT_NAME", "Jarvis").strip() or "Jarvis"
# How the assistant addresses the user in the greeting ("sir", "master", a
# name, ...). Cosmetic; empty hides the form of address.
USER_HONORIFIC = os.environ.get("RAG_USER_HONORIFIC", "sir")
# Interactive chart defaults (used by the chart window). Top-N caps how many
# categories a bar/pie shows before the rest are grouped into "Other".
CHART_TOP_N = int(os.environ.get("RAG_CHART_TOP_N", "20"))
CHART_VALUE_LABELS = os.environ.get("RAG_CHART_LABELS", "1") == "1"  # value labels on bars
# ---- Spark-era features (each independently switchable) -------------------- #
TABLE_CARDS = False
TABLE_CARDS = os.environ.get("RAG_TABLE_CARDS", "1") == "1"
SKIP_TABLE_PROFILING = True  # ← ADD THIS LINE       # per-table semantic cards
SCHEMA_SAMPLES = os.environ.get("RAG_SCHEMA_SAMPLES", "1") == "1"  # sample values in schema block
TIER05_ENABLED = os.environ.get("RAG_TIER05", "1") == "1"          # zero-LLM planning for simple shapes
PLAN_CACHE = os.environ.get("RAG_PLAN_CACHE", "1") == "1"          # normalized question -> bound plan
CONDITION_COVERAGE = os.environ.get("RAG_COVERAGE", "1") == "1"    # no silently dropped conditions
PLAN_V2 = os.environ.get("RAG_PLAN_V2", "1") == "1"                # as-at/having/union/compare/derived
EXPLAIN = os.environ.get("RAG_EXPLAIN", "0") == "1"               # show the decision trace in every answer
# Interpretation clarification (ambiguity.py): when a question term is BOTH a
# stored value AND part of a measure column ('EUR' in CURRENCY vs AMOUNT
# EURO), ask the user which reading they mean -- options phrased from column
# MEANINGS, never column names. Deterministic, 0 LLM calls.
INTERPRET_CLARIFY = os.environ.get("RAG_INTERPRET_CLARIFY", "1") == "1"
# Opt-in: also ask when one value exists in several columns and nothing in
# the question disambiguates (default OFF -- column choice stays with the
# planner, per the long-standing policy in grounding.find_ambiguity).
COLUMN_CLARIFY = os.environ.get("RAG_COLUMN_CLARIFY", "0") == "1"
# When a question cannot be answered (plan rejected, a condition could not be
# applied), let the understanding model PROPOSE one concrete way forward as a
# yes/no question, instead of dead-ending. Model-written, not templated; one
# extra call only on the failure path. Disable with RAG_PROPOSE=0.
PROPOSE_ALTERNATIVES = os.environ.get("RAG_PROPOSE", "1") == "1"
# House rules: one editable text file of standing business policies ("always
# show the company name AND its code", "'in EUR' means converted") injected
# into BOTH the planner prompt and the codegen prompt. Lives next to the
# index so non-developers can maintain it. Empty/absent file = no-op.
HOUSE_RULES_PATH = os.environ.get(
    "RAG_HOUSE_RULES", os.path.join(DATA_DIR, "rules.txt"))
# Words that mean the user ALREADY decided the "use the converted/derived
# measure over ALL rows" reading (so the app should not ask, just enforce
# it). Comma-separated, override-able; matched as whole words / prefixes.
# Empty = the app will ALWAYS ask on a value-vs-measure ambiguity.
INTERPRET_DECIDED_WORDS = [
    w.strip() for w in os.environ.get(
        "RAG_DECIDED_WORDS",
        "equivalent,equiv,converted,conversion,denominated,"
        "ισοδυναμ,μετατρ,εκφρασμεν").split(",") if w.strip()]
# Words that mark a dimension column as "currency-like" -- COSMETIC ONLY
# (picks nicer option wording). The feature behaves identically when this
# matches nothing. Override or empty out freely.
INTERPRET_CURRENCY_WORDS = [
    w.strip() for w in os.environ.get(
        "RAG_CURRENCY_WORDS", "currency,ccy,νομισμα").split(",") if w.strip()]
# When a VALID plan returns zero rows, report the honest zero WITH the plan's
# reasoning instead of silently falling through to free-form codegen (which
# tends to rewrite the logic wrongly). The codegen tier is still used when no
# valid plan could be built at all.
PLAN_ZERO_IS_ANSWER = os.environ.get("RAG_PLAN_ZERO_IS_ANSWER", "1") == "1"


# --------------------------------------------------------------------------- #
# Deterministic-path diagnostics                                              #
# --------------------------------------------------------------------------- #
def dbg(where: str, exc: BaseException) -> None:
    """Surface a swallowed exception from the DETERMINISTIC code paths.

    Those paths narrow their ``except`` clauses to the exception types they
    actually expect (a non-numeric column, a missing key, an unparseable
    date). When one fires it is a normal, recoverable degradation -- but it
    should not be INVISIBLE: a silently-swallowed failure during column
    profiling or grounding quietly produces a worse answer with no trail.

    With RAG_EXPLAIN=1 this prints the degradation to stderr so it shows up
    next to the decision trace; otherwise it is a no-op. It deliberately never
    raises -- diagnostics must not become a new failure mode.
    """
    if not EXPLAIN:
        return
    try:
        import sys
        print(f"[deterministic-degrade] {where}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
    except Exception:
        pass
