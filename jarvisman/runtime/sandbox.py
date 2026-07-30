

from __future__ import annotations

import ast
import builtins as _builtins
import multiprocessing as mp
import queue as _queue
from typing import Any, Callable, Optional

from jarvisman import config as cfg

# Builtins that legitimate plotting/analysis code may use.
_SAFE_BUILTIN_NAMES = {
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "int", "len", "list", "map", "max",
    "min", "pow", "print", "range", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "zip", "abs", "complex",
}

# Attribute calls that reach the filesystem/OS through the PROVIDED objects
# (pd/np/plt are handed in, so 'open' being absent does not stop
# pd.read_csv('/etc/passwd') or pd.read_pickle -- the latter is arbitrary
# code execution by design). None of these is needed by legitimate
# plot/analysis code over the already-loaded tables.
_FORBIDDEN_ATTRS = {
    "read_csv", "read_pickle", "read_excel", "read_parquet", "read_json",
    "read_sql", "read_sql_query", "read_sql_table", "read_table",
    "read_html", "read_feather", "read_hdf", "read_xml", "read_clipboard",
    "to_csv", "to_pickle", "to_excel", "to_parquet", "to_json", "to_sql",
    "to_hdf", "to_feather", "to_clipboard", "to_latex", "to_markdown",
    "savefig", "save", "fromfile", "tofile", "load", "loadtxt",
    "genfromtxt", "savetxt", "memmap", "DataSource",
    "system", "popen", "spawn", "startfile",
}

# Names that must never appear, even via the AST.
_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "input", "__import__", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "exit", "quit",
    "memoryview", "breakpoint", "help", "dir", "type", "object", "super",
    "classmethod", "staticmethod", "property", "__builtins__", "os", "sys",
    "subprocess", "socket", "shutil", "importlib", "open", "file",
}


def validate_code(code: str) -> tuple[bool, str]:
    """Static gate. Returns (is_allowed, reason)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "imports are not allowed (pd, np, plt, dfs, df are already provided)"
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return False, f"access to private/dunder attribute '{node.attr}' is not allowed"
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRS:
            return False, (f"'{node.attr}' is not allowed in the sandbox "
                           "(no file, network or OS access; the data is "
                           "already loaded in df/dfs)")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return False, f"use of '{node.id}' is not allowed"
    return True, "ok"


def _safe_builtins() -> dict:
    b = {n: getattr(_builtins, n) for n in _SAFE_BUILTIN_NAMES if hasattr(_builtins, n)}
    b["__import__"] = None  # belt-and-braces: kill imports at runtime too
    return b


def _apply_child_rlimits() -> None:
    """Cap the child's address space (and CPU time) so a runaway allocation or
    tight loop in generated code cannot take the host down before the
    wall-clock timeout fires.

    POSIX only -- silently skipped where the ``resource`` module is absent
    (Windows, some frozen builds), where the spawn-timeout in
    ``_spawn_and_collect`` remains the backstop. Call this AFTER the heavy
    imports (pandas/numpy/matplotlib) and BEFORE executing user code: the cap
    must never make a legitimate import fail, only the generated code's own
    allocations. Setting a limit must itself never raise -- a host that forbids
    lowering rlimits just keeps the timeout-only behaviour.
    """
    try:
        import resource
    except Exception:
        return  # not POSIX -- the wall-clock timeout is the containment

    def _cap(which, value: int) -> None:
        if not value or value <= 0:
            return
        try:
            _soft, hard = resource.getrlimit(which)
            cap = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(which, (cap, hard))
        except (ValueError, OSError):
            pass  # cannot lower (sandbox host / existing hard limit) -- skip

    mb = getattr(cfg, "SANDBOX_MEM_LIMIT_MB", 0)
    if mb and mb > 0:
        _cap(resource.RLIMIT_AS, int(mb) * 1024 * 1024)
    secs = getattr(cfg, "SANDBOX_CPU_LIMIT_S", 0)
    if secs and secs > 0:
        _cap(resource.RLIMIT_CPU, int(secs))


# --------------------------------------------------------------------------- #
# Plot executor (child process)                                               #
# --------------------------------------------------------------------------- #
def _run_user_code(code: str, dfs: dict, out_q: "mp.Queue") -> None:
    """Executed inside the child process. Renders a figure to PNG bytes."""
    import contextlib
    import io
    import traceback

    try:
        import matplotlib
        matplotlib.use("Agg")  # headless; must precede pyplot import
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except Exception as exc:  # pragma: no cover - environment dependent
        out_q.put({"ok": False, "image": None, "stdout": "", "error": f"sandbox setup failed: {exc}"})
        return

    primary = next(iter(dfs.values())) if dfs else None
    sandbox_globals = {
        "__builtins__": _safe_builtins(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "dfs": dfs,
        "df": primary,
    }

    _apply_child_rlimits()  # cap memory/CPU now that imports are done
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, "<sandboxed-plot>", "exec"), sandbox_globals, sandbox_globals)
        fig = plt.gcf()
        if not fig.get_axes():
            out_q.put({"ok": False, "image": None, "stdout": stdout.getvalue(),
                       "error": "the code ran but produced no plot (figure has no axes)."})
            plt.close("all")
            return
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=cfg.PLOT_DPI, bbox_inches="tight")
        plt.close("all")
        out_q.put({"ok": True, "image": buf.getvalue(), "stdout": stdout.getvalue(), "error": None})
    except Exception:
        out_q.put({"ok": False, "image": None, "stdout": stdout.getvalue(),
                   "error": traceback.format_exc(limit=4)})


# --------------------------------------------------------------------------- #
# Analysis executor (child process)                                           #
# --------------------------------------------------------------------------- #
def _format_result(obj: Any, pd) -> tuple[Optional[str], Optional[str]]:
    """Turn a result object into (plain_text, html_table_or_None). Numeric
    values are rendered with locale-aware separators (numfmt); 'plain' style
    keeps the legacy byte-identical output."""
    from jarvisman.runtime import numfmt
    plain = numfmt.style() == "plain"

    def _fmts(df):
        out = {}
        for col in df.columns:
            if getattr(df[col].dtype, "kind", "O") not in "iuf":
                continue
            if numfmt.is_rate_col(col):
                # rates always display as percentages (4.236%), whichever
                # rate column the generated code selected
                out[col] = numfmt.rate_formatter(df[col].tolist())
            else:
                out[col] = numfmt.fmt
        return out

    if obj is None:
        return None, None
    try:
        if isinstance(obj, pd.DataFrame):
            # Two consumers, two caps: the TEXT is read by the LLM (synthesis)
            # and the eval and must stay small; the HTML goes to the UI, which
            # scrolls, so it may carry far more.
            t_cap = obj.head(cfg.QUERY_MAX_RESULT_ROWS)
            _tbl_cap = getattr(cfg, "QUERY_MAX_TABLE_ROWS", 2000)
            h_cap = obj if _tbl_cap <= 0 else obj.head(_tbl_cap)
            extra = len(obj) - len(t_cap)
            note = "" if extra <= 0 else f"\n... ({extra} more rows)"
            ft = None if plain else _fmts(t_cap)
            fh = None if plain else _fmts(h_cap)
            return (t_cap.to_string(formatters=ft) + note,
                    h_cap.to_html(border=1, index=True, formatters=fh))
        if isinstance(obj, pd.Series):
            t_cap = obj.head(cfg.QUERY_MAX_RESULT_ROWS)
            _tbl_cap = getattr(cfg, "QUERY_MAX_TABLE_ROWS", 2000)
            h_cap = obj if _tbl_cap <= 0 else obj.head(_tbl_cap)
            extra = len(obj) - len(t_cap)
            note = "" if extra <= 0 else f"\n... ({extra} more rows)"
            _sfmt = numfmt.fmt
            if numfmt.is_rate_col(getattr(obj, "name", "")):
                _sfmt = numfmt.rate_formatter(obj.dropna().tolist())
            shown = t_cap if plain or getattr(t_cap.dtype, "kind", "O") \
                not in "iuf" else t_cap.map(_sfmt)
            shown_h = h_cap if plain or getattr(h_cap.dtype, "kind", "O") \
                not in "iuf" else h_cap.map(_sfmt)
            return (shown.to_string() + note,
                    shown_h.to_frame().to_html(border=1, index=True))
        if isinstance(obj, (list, tuple, set)) and len(obj) > 12:
            # A long list rendered as repr text is unreadable and can be
            # megabytes (seen in production: a cross-joined name-change list
            # repeated hundreds of times). Convert to a real table so the
            # normal row caps and big-table gating apply; when the code
            # produced heavy duplication, collapse to unique values + count.
            items = [str(x) for x in obj]
            ser = pd.Series(items, name="value")
            vc = ser.value_counts()
            if len(vc) <= len(ser) * 0.6:
                df2 = vc.rename_axis("value").reset_index(name="count")
            else:
                df2 = ser.to_frame()
            return _format_result(df2, pd)
        if isinstance(obj, (list, tuple, set)):
            # short lists read best as a plain comma line, not a repr
            return ", ".join(str(x) for x in obj), None
        if not plain and isinstance(obj, (int, float)) \
                and not isinstance(obj, bool):
            return numfmt.fmt(obj), None
        text = str(obj)
        if len(text) > 6000:
            text = text[:6000] + f" … (truncated, {len(str(obj)):,} chars)"
        return text, None
    except Exception as exc:
        return f"(result could not be formatted: {exc})", None


def _run_query_code(code: str, dfs: dict, out_q: "mp.Queue") -> None:
    """Executed inside the child process. Computes a `result` and/or stdout."""
    import contextlib
    import io
    import traceback

    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:  # pragma: no cover - environment dependent
        out_q.put({"ok": False, "text": None, "table_html": None, "stdout": "",
                   "error": f"sandbox setup failed: {exc}"})
        return

    primary = next(iter(dfs.values())) if dfs else None
    sandbox_globals = {
        "__builtins__": _safe_builtins(),
        "pd": pd,
        "np": np,
        "dfs": dfs,
        "df": primary,
    }

    _apply_child_rlimits()  # cap memory/CPU now that imports are done
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, "<sandboxed-query>", "exec"), sandbox_globals, sandbox_globals)
        _res_obj = sandbox_globals.get("result")
        text, table_html = _format_result(_res_obj, pd)
        try:
            _rc = int(len(_res_obj)) if hasattr(_res_obj, "__len__") else 1
        except Exception:
            _rc = None
        captured = stdout.getvalue()
        if text is None and not captured.strip():
            out_q.put({"ok": False, "text": None, "table_html": None, "stdout": captured,
                       "error": "the code ran but did not assign `result` or print anything."})
            return
        out_q.put({"ok": True, "text": text, "table_html": table_html, "row_count": _rc, "stdout": captured, "error": None})
    except Exception:
        out_q.put({"ok": False, "text": None, "table_html": None, "stdout": stdout.getvalue(),
                   "error": traceback.format_exc(limit=4)})


# --------------------------------------------------------------------------- #
# Shared spawn/timeout harness                                                #
# --------------------------------------------------------------------------- #
def _spawn_and_collect(target: Callable, code: str, dfs: dict, timeout: int) -> dict:
    ctx = mp.get_context("spawn")  # never 'fork' from a threaded Qt process
    out_q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(code, dfs, out_q), daemon=True)
    proc.start()
    try:
        # Drain the result BEFORE join to avoid the Queue feeder deadlock.
        result = out_q.get(timeout=timeout)
    except _queue.Empty:
        result = {"ok": False, "image": None, "text": None, "table_html": None, "stdout": "",
                  "error": f"execution timed out after {timeout}s and was terminated."}
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=3)
    return result


def run_sandboxed(code: str, dfs: dict, timeout: int = cfg.SANDBOX_TIMEOUT) -> dict:
    """Validate then execute *plotting* code; returns a dict with PNG bytes."""
    allowed, reason = validate_code(code)
    if not allowed:
        return {"ok": False, "image": None, "stdout": "", "error": f"rejected by sandbox: {reason}"}
    return _spawn_and_collect(_run_user_code, code, dfs, timeout)


def run_query(code: str, dfs: dict, timeout: int = cfg.SANDBOX_TIMEOUT) -> dict:
    """Validate then execute *analysis* code; returns a dict with text/table."""
    allowed, reason = validate_code(code)
    if not allowed:
        return {"ok": False, "text": None, "table_html": None, "stdout": "",
                "error": f"rejected by sandbox: {reason}"}
    return _spawn_and_collect(_run_query_code, code, dfs, timeout)