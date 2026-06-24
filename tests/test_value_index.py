"""Locks the deterministic name-resolution behaviour that real usage hardened:
free-typed names map to their stored 'Surname,First' spelling, the documented
one- and two-typo cases keep working, and an ambiguous/nonsense phrase is
NEVER auto-picked. The 0.78->0.6 comment drift bug lives here as an explicit
invariant so nobody silently re-tightens the bar and breaks the two-typo case.
"""
from __future__ import annotations

from jarvisman.semantics import value_index as vi
from jarvisman.semantics.value_index import resolve_phrase, resolve_phrase_fuzzy


def _resolve(vindex, phrase):
    return resolve_phrase(vindex, phrase) or resolve_phrase_fuzzy(vindex, phrase)


def test_exact_token_reorder_resolves(vindex):
    # typed first-last; stored 'Surname,First' -- pure token match, no typo
    assert _resolve(vindex, "Panagiotis Karanikolaos") == "Karanikolaos,Panagiotis"


def test_single_typo_resolves(vindex):
    # 'Spurou' for 'Spyrou', reversed order
    assert _resolve(vindex, "Spiros Spurou") == "Spyrou,Spiros"


def test_two_typo_resolves(vindex):
    # the hard documented case: 'spuros' vs 'spyrou' has ratio 0.667, so the
    # token bar MUST stay <= 0.667 for this to resolve
    assert _resolve(vindex, "Spuros spirou") == "Spyrou,Spiros"


def test_nonsense_does_not_autoresolve(vindex):
    assert _resolve(vindex, "Xavier Yolanda") is None
    assert _resolve(vindex, "Spxxxx Yyyyy") is None


def test_fuzzy_token_ratio_invariant():
    # Regression guard for the comment/code drift: the documented two-typo case
    # needs ratio <= 0.667. A stricter bar silently breaks it.
    assert vi._FUZZY_TOKEN_RATIO <= 0.667


def test_resolve_phrase_exact_value_returns_none(vindex):
    # a phrase that already exists verbatim must not be "resolved" to itself
    assert resolve_phrase(vindex, "Spyrou,Spiros") is None
