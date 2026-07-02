import os, re, sys
os.environ.setdefault("RAG_EXPLAIN", "1")  # capture traces on failures

from jarvisman import config as cfg
from jarvisman.llm.ollama_client import OllamaClient
from jarvisman.retrieval.vector_store import VectorStore
from jarvisman.retrieval.rag import RAGPipeline
from jarvisman.agent import Agent

def load_cases(path):
    cases, cur = [], {}
    for line in open(path, encoding="utf-8"):
        s = line.rstrip("\n")
        if s.strip().startswith("#") or not s.strip():
            continue
        if s == "---":
            if cur: cases.append(cur); cur = {}
        elif s.startswith("Q:"): cur["q"] = s[2:].strip()
        elif s.startswith("A:"): cur["a"] = s[2:].strip()
        elif s.startswith("TYPE:"): cur["type"] = s[5:].strip()
    if cur: cases.append(cur)
    return cases

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
def nums(t): return [float(x.replace(",", "")) for x in _NUM.findall(t or "")]

def check(case, text):
    typ = case.get("type", "contains")
    if typ == "exact_number":
        want = nums(case["a"])
        got = nums(text)
        if not want: return False, "no expected number"
        w = want[0]
        for g in got:
            if abs(g - w) <= max(abs(w) * 1e-4, 0.01):
                return True, f"matched {g:,.2f}"
        return False, f"expected {w:,.2f}, got {got[:5]}"
    if typ == "contains":
        ok = case["a"].lower() in (text or "").lower()
        return ok, "found" if ok else "expected substring not present"
    return None, "shape check — inspect manually"

def main():
    cases = load_cases("eval_questions.txt")
    oll = OllamaClient(cfg.OLLAMA_HOST); store = VectorStore()
    rag = RAGPipeline(oll, store, cfg.DEFAULT_CHAT_MODEL, cfg.DEFAULT_EMBED_MODEL)
    dfs = rag.load_persisted(cfg.INDEX_DIR)
    agent = Agent(oll, rag, cfg.DEFAULT_CHAT_MODEL); agent.dataframes = dfs

    passed = 0; manual = 0
    for i, c in enumerate(cases, 1):
        res = agent.handle(c["q"])
        text = res.get("text") or ""
        ok, why = check(c, text)
        tool = res.get("tool")
        if ok is True:
            passed += 1; print(f"[{i:2}] PASS  ({tool})  {why}")
        elif ok is None:
            manual += 1; print(f"[{i:2}] MANUAL({tool})  {c['q'][:60]}")
        else:
            print(f"[{i:2}] FAIL  ({tool})  {why}")
            print(f"       Q: {c['q'][:70]}")
            tr = res.get("trace") or []
            for t in (tr[-4:] if tr else []): print(f"       trace: {t}")
            print(f"       got: {text[:120]}")
    auto = len(cases) - manual
    print(f"\nSCORE: {passed}/{auto} auto-checked pass" + (f"  ({manual} need manual review)" if manual else ""))

if __name__ == "__main__":
    main()
