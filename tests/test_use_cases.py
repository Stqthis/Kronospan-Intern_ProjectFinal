"""Pytest wrapper for the use-case regression harness.

Runs the ground-truth checks when a data folder is available (set
RAG_UCTEST_DATA to the folder with the Kronospan files); otherwise skips, so
CI stays green without the proprietary data. Run the full report directly with:
    RAG_UCTEST_DATA=/path/to/files python -m eval.run_use_cases
"""
import os
import glob

import pytest

DATA_DIR = os.environ.get("RAG_UCTEST_DATA", "")


def _have(*patterns):
    if not DATA_DIR or not os.path.isdir(DATA_DIR):
        return None
    for pat in patterns:
        hits = [h for h in glob.glob(os.path.join(DATA_DIR, "**", pat), recursive=True)
                if "~$" not in os.path.basename(h)]
        if hits:
            return sorted(hits)[0]
    return None


@pytest.mark.skipif(not DATA_DIR, reason="set RAG_UCTEST_DATA to run use-case checks")
def test_cy01_funds():
    from eval.run_use_cases import check_cy01
    path = _have("CY01-DDR*DATA*.xlsx", "CY01*DATA*.xlsx")
    if not path:
        pytest.skip("CY01 data file not present")
    for name, ok, detail in check_cy01(path):
        assert ok, f"{name}: {detail}"


@pytest.mark.skipif(not DATA_DIR, reason="set RAG_UCTEST_DATA to run use-case checks")
def test_cy05_directorships_and_address():
    from eval.run_use_cases import check_cy05
    path = _have("CY05*DATA*.xlsx", "CY05*Group*Company*.xlsx")
    if not path:
        pytest.skip("CY05 data file not present")
    for name, ok, detail in check_cy05(path):
        assert ok, f"{name}: {detail}"
