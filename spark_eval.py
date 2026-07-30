#!/usr/bin/env python3
"""
spark_eval.py -- measure REAL end-to-end accuracy on the DGX Spark, with the
actual llama3:70b + qwen3:30b. This is the ONLY honest accuracy number: it runs
your real questions through the real models and checks the answers.

USAGE
  1. Index your real Excel/PDF files in the app once.
  2. Edit questions.jsonl: one {"q": ..., "expect": ...} per line, where
     'expect' is a string/number that MUST appear in a correct answer (or a
     list of strings that must ALL appear). Write questions as a NON-expert
     user would -- general business language, not column names.
  3. Run:  RAG_CODE_MODEL=qwen3:30b RAG_UNDERSTANDING_MODEL=llama3:70b \
           RAG_EXPLAIN=1 python spark_eval.py questions.jsonl
  4. Read accuracy + the per-question trace (which table/columns the model
     chose). Send me the failures and I fix the engine/prompt, not the number.

It logs, per question: chosen table, chosen columns, whether the expected
value appeared, latency, and the decision trace. Nothing is mocked.
"""
import sys, json, time, os

def load_app():
    # build the same pipeline the GUI uses
    from jarvisman import config as cfg
    from jarvisman.llm.ollama_client import OllamaClient
    from jarvisman.retrieval.rag import RAGPipeline
    from jarvisman.retrieval.vector_store import VectorStore
    from jarvisman.agent import Agent
    data_dir = os.environ.get("RAG_DATA_DIR", os.getcwd())
    index_dir = cfg.INDEX_DIR
    client = OllamaClient()
    chat_model = os.environ.get("RAG_CHAT_MODEL", "qwen3:30b")
    rag = RAGPipeline(client, VectorStore(), chat_model,
                      os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text"))
    # load a previously-built index exactly as the app does
    try:
        tables = rag.load_persisted(index_dir)
    except Exception as e:
        print(f"ERROR: could not load index from {index_dir}: {e}")
        print("Build the index in the app first (same machine), or set RAG_INDEX_DIR.")
        sys.exit(1)
    from jarvisman.semantics.value_index import ValueIndex
    agent = Agent(client, rag, chat_model)
    agent.dataframes = tables
    agent.value_index = ValueIndex.build(tables)
    return agent

def expect_ok(answer, expect):
    a = (answer or "").lower().replace(",", "")
    if isinstance(expect, list):
        return all(str(e).lower().replace(",", "") in a for e in expect)
    return str(expect).lower().replace(",", "") in a

def main():
    if len(sys.argv) < 2:
        print("usage: python spark_eval.py questions.jsonl"); return
    agent = load_app()
    cases = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    P = F = 0; rows = []
    for i, c in enumerate(cases, 1):
        q = c["q"]; expect = c.get("expect", "")
        t0 = time.time()
        try:
            res = agent.handle(q)
            ans = res.get("text", "") or ""
            trace = res.get("trace") or []
        except Exception as e:
            ans = f"ERROR: {e}"; trace = []
        dt = time.time() - t0
        ok = expect_ok(ans, expect) if expect else None
        if ok is True: P += 1
        elif ok is False: F += 1
        mark = "OK " if ok else ("XX " if ok is False else "?? ")
        print(f"{mark}[{i}/{len(cases)}] ({dt:.1f}s) {q[:60]}")
        if ok is False:
            print(f"     expected: {expect!r}")
            print(f"     got:      {ans[:120]!r}")
            if trace: print(f"     trace:    {' | '.join(str(t) for t in trace[:4])}")
        rows.append({"q": q, "expect": expect, "ok": ok, "answer": ans[:300],
                     "trace": trace, "seconds": round(dt, 1)})
    graded = P + F
    print(f"\n{'='*60}")
    if graded:
        print(f"REAL END-TO-END ACCURACY: {P}/{graded} = {P/graded*100:.1f}%")
    print(f"avg latency: {sum(r['seconds'] for r in rows)/len(rows):.1f}s/question")
    json.dump(rows, open("spark_eval_results.json", "w"), indent=2, ensure_ascii=False)
    print("full results -> spark_eval_results.json (send me this file)")

if __name__ == "__main__":
    main()
