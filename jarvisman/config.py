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
    given chat_model when the role-specific model is not configured."""
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
CHUNK_SIZE = 100
CHUNK_OVERLAP = 100
RAG_TOP_K = 5

# Table profiling — accuracy mode. Profiling builds the per-column meaning map
# the reasoner uses to bind questions to the right column. Do NOT disable.
SKIP_TABLE_PROFILING = False
TABLE_CARDS = True
TABLE_PROFILING_ROWS = 2000   # profile a sample of rows (not all, not zero)

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
CATEGORICAL_MAX_UNIQUE = 200
ANALYSIS_MAX_RETRIES = 2

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
NUM_CTX = int(os.environ.get("RAG_NUM_CTX", "8192"))

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
PLAN_REPAIR_ATTEMPTS = 1
SYNTHESIZE_ANSWER = os.environ.get("RAG_SYNTH", "0") == "1"
SYNTHESIZE_MAX_ROWS = int(os.environ.get("RAG_SYNTH_MAX_ROWS", "3"))
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