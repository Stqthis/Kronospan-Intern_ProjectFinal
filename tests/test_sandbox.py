"""The sandbox is the one place that executes model-written code, so its two
guarantees are locked here: (1) the static AST gate rejects anything that
could touch the filesystem/OS/imports, and (2) the child resource caps contain
a runaway allocation instead of letting it OOM the host.

These run real subprocesses via multiprocessing-`spawn`; they are slightly
slower than the pure-function tests but they are the security boundary.
"""
from __future__ import annotations

import pandas as pd
import pytest

from jarvisman.runtime import sandbox


# --------------------------------------------------------------------------- #
# Static AST gate                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "from os import system",
        "result = open('/etc/passwd').read()",
        "result = ().__class__.__bases__",          # dunder escape attempt
        "result = pd.read_csv('/etc/passwd')",      # file read via provided obj
        "result = df.to_csv('/tmp/x.csv')",         # file write
        "plt.savefig('/tmp/x.png')",                # figure exfiltration
        "result = eval('1+1')",
        "result = __import__('os').getcwd()",
        "result = getattr(df, 'to_pickle')('x')",   # getattr indirection
    ],
)
def test_validate_code_rejects_dangerous(code):
    ok, reason = sandbox.validate_code(code)
    assert not ok, f"should have been rejected: {code}"
    assert reason and reason != "ok"


@pytest.mark.parametrize(
    "code",
    [
        "result = df.groupby('COMPANY')['AMOUNT EURO'].sum()",
        "result = df[df['CURRENCY'] == 'EUR']",
        "result = int(len(df))",
        "result = df['AMOUNT EURO'].mean()",
    ],
)
def test_validate_code_allows_legit(code):
    ok, _ = sandbox.validate_code(code)
    assert ok, f"legit analysis code should pass: {code}"


# --------------------------------------------------------------------------- #
# Execution + containment                                                    #
# --------------------------------------------------------------------------- #
def _dfs():
    return {
        "f.xlsx:S": pd.DataFrame(
            {"COMPANY": ["A", "B", "A"], "AMOUNT EURO": [10, 20, 30]}
        )
    }


def test_run_query_legit():
    code = "result = df.groupby('COMPANY')['AMOUNT EURO'].sum().reset_index()"
    r = sandbox.run_query(code, _dfs(), timeout=20)
    assert r["ok"], r.get("error")
    assert "40" in (r.get("text") or "")  # A = 10 + 30


def test_run_query_rejected_code_never_executes():
    r = sandbox.run_query("import os; result = os.listdir('/')", _dfs(), timeout=20)
    assert not r["ok"]
    assert "rejected by sandbox" in (r.get("error") or "")


def _has_resource():
    try:
        import resource  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_resource(), reason="POSIX rlimits unavailable")
def test_memory_bomb_is_contained():
    """A ~38 GB virtual allocation must be killed by RLIMIT_AS (default 4 GB),
    surfacing as a MemoryError -- not an OOM that takes down the host."""
    bomb = "result = np.ones((600_000_000, 8), dtype='float64').sum()"
    r = sandbox.run_query(bomb, _dfs(), timeout=25)
    assert not r["ok"]
    assert "memory" in (r.get("error") or "").lower(), r.get("error")
