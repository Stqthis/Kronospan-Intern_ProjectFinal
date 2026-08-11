"""Headless command-line interface: index files and ask questions with no GUI.

This is what makes the project usable inside Docker, over SSH, and for batch
work (indexing hundreds of files from a folder tree). It drives the exact same
pipeline as the desktop app -- same ingestion, same reasoner, same persisted
index -- so anything indexed here is immediately available in the GUI and
vice versa.

Usage:
    python -m jarvisman.cli index <file-or-folder> [...]   # build/update index
    python -m jarvisman.cli ask "question"                  # one-shot answer
    python -m jarvisman.cli chat                             # interactive REPL
    python -m jarvisman.cli status                           # what is indexed
    python -m jarvisman.cli check --data <folder>            # ground-truth eval
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

from jarvisman import config as cfg


def _collect_paths(items: list[str]) -> list[str]:
    """Expand folders (recursively) and globs into a flat list of supported
    files, skipping Office lock files (~$...)."""
    out: list[str] = []
    for item in items:
        if os.path.isdir(item):
            for ext in cfg.SUPPORTED_EXTENSIONS:
                out.extend(glob.glob(os.path.join(item, "**", f"*{ext}"),
                                     recursive=True))
        else:
            hits = glob.glob(item)
            out.extend(hits if hits else [item])
    seen, files = set(), []
    for p in out:
        base = os.path.basename(p)
        if base.startswith("~$"):
            continue
        if not p.lower().endswith(cfg.SUPPORTED_EXTENSIONS):
            continue
        rp = os.path.abspath(p)
        if rp not in seen:
            seen.add(rp)
            files.append(rp)
    return files


def _build_stack():
    from jarvisman.llm.ollama_client import OllamaClient
    from jarvisman.retrieval.vector_store import VectorStore
    from jarvisman.retrieval.rag import RAGPipeline
    from jarvisman.agent import Agent

    ollama = OllamaClient(cfg.OLLAMA_HOST)
    store = VectorStore()
    rag = RAGPipeline(ollama, store, cfg.DEFAULT_CHAT_MODEL,
                      cfg.DEFAULT_EMBED_MODEL)
    agent = Agent(ollama, rag, cfg.DEFAULT_CHAT_MODEL)
    return ollama, store, rag, agent


def _load_index(rag, agent) -> bool:
    try:
        tables = rag.load_persisted(cfg.INDEX_DIR)
        agent.dataframes = tables or {}
        return True
    except FileNotFoundError:
        return False


def cmd_index(args) -> int:
    files = _collect_paths(args.paths)
    if not files:
        print("No supported files found "
              f"(looking for {', '.join(cfg.SUPPORTED_EXTENSIONS)}).")
        return 2
    print(f"Indexing {len(files)} file(s) ...")
    _, _, rag, agent = _build_stack()
    # incremental: previously indexed files are kept unless re-indexed
    try:
        rag.load_persisted(cfg.INDEX_DIR)
    except FileNotFoundError:
        pass
    stats, dataframes = rag.index_documents(
        files, progress_callback=lambda m: print(f"  {m}"))
    agent.dataframes = dataframes
    print(f"\nDone: {stats['pdf_chunks']} text chunk(s), "
          f"{stats['tables']} table(s) now queryable.")
    for k, df in list(dataframes.items())[:20]:
        print(f"  {k}  ({df.shape[0]}\u00d7{df.shape[1]})")
    if len(dataframes) > 20:
        print(f"  ... and {len(dataframes) - 20} more")
    return 0


def _print_answer(res: dict) -> None:
    text = (res or {}).get("text") or ""
    if text:
        print(text)
    th = (res or {}).get("table_html")
    if th and not text:
        import re
        rows = re.sub(r"</tr>", "\n", th)
        rows = re.sub(r"</t[dh]>", "\t", rows)
        rows = re.sub(r"<[^>]+>", "", rows)
        print(rows.strip())
    opts = (res or {}).get("options") or []
    if opts:
        print("\nOptions:")
        for i, o in enumerate(opts, 1):
            print(f"  {i}. {o}")


def cmd_ask(args) -> int:
    _, _, rag, agent = _build_stack()
    if not _load_index(rag, agent):
        print("No saved index found. Run:  python -m jarvisman.cli index <files>")
        return 2
    res = agent.handle(args.question)
    _print_answer(res)
    return 0


def cmd_chat(args) -> int:
    _, _, rag, agent = _build_stack()
    if not _load_index(rag, agent):
        print("No saved index found. Run:  python -m jarvisman.cli index <files>")
        return 2
    print(f"{len(agent.dataframes)} table(s) loaded. "
          "Type a question, or 'exit' to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            return 0
        res = agent.handle(q)
        _print_answer(res)


def cmd_status(args) -> int:
    _, store, rag, agent = _build_stack()
    if not _load_index(rag, agent):
        print("No saved index.")
        return 0
    print(f"Index version : {getattr(rag, 'index_version', '?')}")
    print(f"Text chunks   : {store.count}")
    print(f"Tables        : {len(agent.dataframes)}")
    for k, df in agent.dataframes.items():
        print(f"  {k}  ({df.shape[0]}\u00d7{df.shape[1]})")
    return 0


def cmd_check(args) -> int:
    from eval.run_use_cases import run
    return run(args.data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="jarvisman",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="index files/folders (incremental)")
    p.add_argument("paths", nargs="+")
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("ask", help="answer one question against the saved index")
    p.add_argument("question")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("chat", help="interactive question loop")
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("status", help="show what is indexed")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("check", help="run the ground-truth use-case checks")
    p.add_argument("--data", default=os.environ.get("RAG_UCTEST_DATA", "./demo_data"))
    p.set_defaults(fn=cmd_check)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
