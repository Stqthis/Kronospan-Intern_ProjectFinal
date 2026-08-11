"""HTTP backend: the WHOLE pipeline (index + agent + Ollama access) served
as a JSON API, so the model runs once — in Docker on the DGX — and any
front-end (the desktop app, curl, scripts) is a thin client.

Standard library only (ThreadingHTTPServer): no new dependencies, runs in
the existing Docker image.

    python -m jarvisman.server                    # 0.0.0.0:8800
    RAG_API_PORT=8800 RAG_DATA_DIR=/data ...

Endpoints (all JSON):
    GET  /health          {ok, ollama, model, tables, chunks}
    GET  /status          tables/docs inventory + model + index location
    POST /ask             {"q": "..."} -> the agent's full result dict
    GET  /charts          dashboard chart specs (labels/values, tiny payload)
    GET  /reports         deterministic report catalogue
    POST /report          {"id": N} -> {title, sub, columns, rows}
    POST /clear_history   start a fresh conversation

Answers are computed by the SAME Agent class the desktop app uses, so
accuracy is identical by construction; one request is served at a time
(a lock keeps conversation history coherent — Ollama serializes anyway).
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from jarvisman import config as cfg
from jarvisman.runtime import audit, scoreboard

_STATE: dict = {"agent": None, "rag": None, "store": None, "ollama": None,
                "lock": threading.Lock(), "reports": [], "warnings": []}

# Optional shared-secret auth: when RAG_API_KEY is set every endpoint except
# / (the web page) and /health requires the X-Api-Key header.
_API_KEY = os.environ.get("RAG_API_KEY", "").strip()

# Watch folder: new/changed workbooks and PDFs dropped here are ingested
# automatically (the monthly data-drop contract needs no manual step).
_WATCH_DIR = os.environ.get("RAG_WATCH_DIR", "/files")
_WATCH_INTERVAL = int(os.environ.get("RAG_WATCH_INTERVAL", "60"))
_WATCH_STATE_PATH = os.path.join(cfg.DATA_DIR, "ingest_watch.json")


# --------------------------------------------------------------------------- #
# boot                                                                         #
# --------------------------------------------------------------------------- #
def _boot() -> None:
    from jarvisman.agent import Agent
    from jarvisman.llm.ollama_client import OllamaClient
    from jarvisman.retrieval.rag import RAGPipeline
    from jarvisman.retrieval.vector_store import VectorStore

    ollama = OllamaClient(cfg.OLLAMA_HOST)
    store = VectorStore()
    rag = RAGPipeline(ollama, store, cfg.DEFAULT_CHAT_MODEL,
                      cfg.DEFAULT_EMBED_MODEL)
    print(f"[server] loading index from {cfg.INDEX_DIR} ...")
    dfs = {}
    try:
        dfs = rag.load_persisted(cfg.INDEX_DIR) or {}
    except Exception as exc:
        print(f"[server] no persisted index loaded ({type(exc).__name__}: {exc})")
    agent = Agent(ollama, rag, cfg.DEFAULT_CHAT_MODEL)
    agent.dataframes = dfs
    _STATE.update(agent=agent, rag=rag, store=store, ollama=ollama)
    print(f"[server] {len(dfs)} table(s), {store.count} chunk(s) ready")

    def _warm():
        models = []
        for role in ("chat", "plan", "codegen", "understand"):
            try:
                m = cfg.model_for(role, agent.chat_model)
            except Exception:
                m = ""
            if m and m not in models:
                models.append(m)
        for m in models:
            try:
                ollama.warmup(m)
                print(f"[server] warmed {m}")
            except Exception:
                pass
    threading.Thread(target=_warm, daemon=True).start()
    threading.Thread(target=_watch_loop, daemon=True).start()


def _warn(msg: str) -> None:
    print(f"[server] WARNING: {msg}")
    _STATE["warnings"].append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "message": msg})
    _STATE["warnings"] = _STATE["warnings"][-50:]


def _validate_dates(path: str) -> None:
    try:
        from jarvisman.ingest.date_check import check
        res = check(path)
    except Exception:
        return
    if res and res.get("mismatch"):
        _warn(f"{res['file']}: file name says {res['filename_date']} but the "
              f"data inside is dated {res['content_date']} — 'as at' "
              f"questions for these dates may answer from the wrong file. "
              f"Rename the export and re-index it.")


def _watch_signatures() -> dict:
    try:
        with open(_WATCH_STATE_PATH, encoding="utf-8") as fh:
            return json.load(fh) or {}
    except Exception:
        return {}


def _watch_loop() -> None:
    """Auto-ingest the data-drop folder: any new or changed supported file
    is validated (filename vs content dates) and indexed incrementally."""
    if not _WATCH_DIR or not os.path.isdir(_WATCH_DIR):
        print(f"[server] watch folder disabled ({_WATCH_DIR!r} not found)")
        return
    print(f"[server] watching {_WATCH_DIR} every {_WATCH_INTERVAL}s")
    seen = _watch_signatures()
    first_pass = not seen
    while True:
        try:
            batch = []
            for root, _dirs, files in os.walk(_WATCH_DIR):
                for f in files:
                    if f.startswith("~$") or not f.lower().endswith(
                            tuple(cfg.SUPPORTED_EXTENSIONS)):
                        continue
                    fp = os.path.join(root, f)
                    try:
                        st = os.stat(fp)
                        sig = f"{int(st.st_mtime)}:{st.st_size}"
                    except OSError:
                        continue
                    if seen.get(fp) != sig:
                        seen[fp] = sig
                        batch.append(fp)
            if batch:
                for fp in batch:
                    _validate_dates(fp)
                if first_pass:
                    # startup pass records what is already there (it is in
                    # the index) and only surfaces date warnings
                    print(f"[server] watch: {len(batch)} existing file(s) "
                          f"registered")
                else:
                    print(f"[server] watch: ingesting {len(batch)} file(s): "
                          + ", ".join(os.path.basename(b) for b in batch))
                    rag, agent = _STATE["rag"], _STATE["agent"]
                    with _STATE["lock"]:
                        try:
                            _stats, dfs = rag.index_documents(batch)
                            agent.dataframes = dfs
                            print(f"[server] watch: index updated — "
                                  f"{len(dfs)} table(s) total")
                        except Exception as exc:
                            _warn(f"auto-ingestion failed for {batch}: "
                                  f"{type(exc).__name__}: {exc}")
                try:
                    with open(_WATCH_STATE_PATH, "w", encoding="utf-8") as fh:
                        json.dump(seen, fh)
                except Exception:
                    pass
                first_pass = False
        except Exception as exc:
            print(f"[server] watch loop error: {type(exc).__name__}: {exc}")
        time.sleep(max(10, _WATCH_INTERVAL))


# --------------------------------------------------------------------------- #
# pure data helpers (no Qt — mirrors of the dashboard/report logic)            #
# --------------------------------------------------------------------------- #
_AMOUNT_HINTS = ("eur", "amount", "balance", "value", "total", "outstanding",
                 "sum", "funds", "capital", "deposit")
_CAT_HINTS = ("country", "currency", "bank", "company", "lender", "borrower",
              "name", "type", "category", "counterparty", "entity")


def _numeric_cols(df):
    return [c for c in df.columns
            if str(df[c].dtype).startswith(("int", "float"))]


def _best_amount_col(df):
    nums = _numeric_cols(df)
    if not nums:
        return None
    for c in nums:
        if any(h in str(c).lower() for h in _AMOUNT_HINTS):
            return c
    try:
        return max(nums, key=lambda c: abs(float(df[c].sum())))
    except Exception:
        return nums[0]


def _cat_cols(df, max_unique: int = 60):
    out = []
    for c in df.columns:
        if str(df[c].dtype) in ("object", "category", "string"):
            try:
                u = df[c].nunique(dropna=True)
            except Exception:
                continue
            if 2 <= u <= max_unique:
                out.append((c, u))
    out.sort(key=lambda cu: (not any(h in str(cu[0]).lower()
                                     for h in _CAT_HINTS), cu[1]))
    return [c for c, _u in out]


def _indexed_documents() -> list:
    counts: dict = {}
    store = _STATE["store"]
    try:
        for c in getattr(store, "chunks", []) or []:
            src = c.get("source")
            if src:
                counts.setdefault(src, set()).add(c.get("location", ""))
    except Exception:
        return []
    return sorted((k, len(v)) for k, v in counts.items())


def _status() -> dict:
    agent, store, ollama = _STATE["agent"], _STATE["store"], _STATE["ollama"]
    tables = [{"name": str(k), "rows": int(df.shape[0]),
               "cols": int(df.shape[1])}
              for k, df in (agent.dataframes or {}).items()]
    docs = [{"file": f, "pages": p} for f, p in _indexed_documents()]
    alive = False
    try:
        alive = bool(ollama.is_alive())
    except Exception:
        pass
    return {"ok": True, "ollama": alive,
            "model": agent.chat_model or cfg.DEFAULT_CHAT_MODEL,
            "tables": tables, "docs": docs,
            "chunks": int(getattr(store, "count", 0)),
            "index_dir": cfg.INDEX_DIR,
            "warnings": _STATE["warnings"][-20:]}


def _chart_specs() -> list:
    """(kind, title, sub, labels, values) — same shapes the dashboard draws."""
    agent = _STATE["agent"]
    tables = agent.dataframes or {}
    docs = _indexed_documents()
    specs, pairs, seen = [], [], set()
    for name, df in tables.items():
        num = _best_amount_col(df)
        if not num:
            continue
        for cat in _cat_cols(df)[:2]:
            pairs.append((str(name), df, str(cat), str(num)))
    for tname, df, cat, num in pairs:
        if cat.lower() in seen or len(specs) >= 3:
            continue
        seen.add(cat.lower())
        try:
            g = df.groupby(cat)[num].sum().sort_values(ascending=False)
        except Exception:
            continue
        top = g.head(8 if len(specs) == 0 else 10)
        labels = [str(i) for i in top.index]
        values = [float(v) for v in top.values]
        if not values or not any(values):
            continue
        kind = "donut" if len(specs) == 0 else (
            "hbar" if len(specs) == 1 else "bar")
        specs.append({"kind": kind, "title": f"{num} by {cat}",
                      "sub": f"{tname} · top {len(labels)}",
                      "labels": labels, "values": values})
    if tables and len(specs) < 4:
        items = sorted(((str(k), int(df.shape[0]))
                        for k, df in tables.items()), key=lambda kv: -kv[1])[:8]
        specs.append({"kind": "donut" if not specs else "bar",
                      "title": "Rows by table", "sub": "loaded tables",
                      "labels": [k for k, _v in items],
                      "values": [v for _k, v in items]})
    if docs and len(specs) < 4:
        top = sorted(docs, key=lambda fp: -fp[1])[:10]
        specs.append({"kind": "hbar", "title": "Document knowledge base",
                      "sub": "indexed pages per PDF",
                      "labels": [f for f, _p in top],
                      "values": [float(p) for _f, p in top]})
    return specs[:4]


def _report_specs() -> list:
    """Rebuilt on demand: [(title, sub, fn)] — same catalogue as the app."""
    import pandas as pd
    agent = _STATE["agent"]
    tables = agent.dataframes or {}
    specs = []
    if tables:
        def _inventory():
            rows = [(f"T{i+1:02d}", str(n), df.shape[0], df.shape[1])
                    for i, (n, df) in enumerate(tables.items())]
            return pd.DataFrame(rows, columns=["CODE", "TABLE",
                                               "ROWS", "COLUMNS"])
        specs.append(("Tables inventory",
                      "every loaded table with its size", _inventory))
    for name, df in tables.items():
        n = str(name)
        specs.append((f"Overview — {n}"[:60],
                      f"first 200 rows of {df.shape[0]:,}",
                      lambda d=df: d.head(200).copy()))
        num = _best_amount_col(df)
        if num:
            for cat in _cat_cols(df)[:2]:
                def _agg(d=df, c=cat, m=num):
                    g = (d.groupby(c)[m].agg(["sum", "count"])
                          .sort_values("sum", ascending=False).reset_index())
                    g.columns = [str(c), f"total {m}", "rows"]
                    return g
                specs.append((f"{num} by {cat}"[:60],
                              f"{n} · grouped totals", _agg))
        if _numeric_cols(df):
            def _summary(d=df):
                s = d.describe().transpose().reset_index()
                return s.rename(columns={"index": "column"}).round(2)
            specs.append((f"Numeric summary — {n}"[:60],
                          "count / mean / min / max per column", _summary))
    docs = _indexed_documents()
    if docs:
        def _docs_report():
            return pd.DataFrame(docs, columns=["FILE", "PAGES"])
        specs.append(("Indexed documents",
                      "PDF knowledge base contents", _docs_report))
    return specs


def _run_report(idx: int) -> dict:
    specs = _report_specs()
    if not 0 <= idx < len(specs):
        return {"error": f"no report {idx}"}
    title, sub, fn = specs[idx]
    df = fn()
    df = df.head(1000)
    cols = [str(c) for c in df.columns]
    rows = [[(None if v != v else v) if isinstance(v, float) else
             (str(v) if not isinstance(v, (int, float, bool)) else v)
            for v in rec] for rec in df.itertuples(index=False, name=None)]
    return {"title": title, "sub": sub, "columns": cols, "rows": rows}


def _json_safe(obj):
    """Deep-copy a result dict into JSON-serializable form (bytes -> b64)."""
    if isinstance(obj, bytes):
        return {"__b64__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj if obj == obj and abs(obj) != float("inf") else None
    return str(obj)


# --------------------------------------------------------------------------- #
# HTTP plumbing                                                                #
# --------------------------------------------------------------------------- #
def _rules_path() -> str:
    return getattr(cfg, "HOUSE_RULES_PATH", "") or os.path.join(
        cfg.DATA_DIR, "rules.txt")


def _rules_text() -> str:
    p = _rules_path()
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as fh:
                return fh.read()
        except Exception:
            return ""
    # no user file yet: show the shipped defaults as the starting point
    try:
        from jarvisman.runtime.house_rules import _DEFAULT_RULES_PATH
        with open(_DEFAULT_RULES_PATH, encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        return ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _authorized(self) -> bool:
        if not _API_KEY:
            return True
        return (self.headers.get("X-Api-Key") or "") == _API_KEY

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception:
            return {}

    def log_message(self, fmt, *args):                    # quieter logs
        print("[server]", self.address_string(), fmt % args)

    def do_GET(self):                                     # noqa: N802
        agent, store = _STATE["agent"], _STATE["store"]
        if self.path in ("/", "/ui", "/index.html"):
            body = _WEB_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if not self.path.startswith("/health") and not self._authorized():
            self._send(401, {"error": "missing or wrong X-Api-Key"})
            return
        if self.path.startswith("/health"):
            alive = False
            try:
                alive = bool(_STATE["ollama"].is_alive())
            except Exception:
                pass
            self._send(200, {"ok": True, "ollama": alive,
                             "model": agent.chat_model,
                             "tables": len(agent.dataframes or {}),
                             "chunks": int(getattr(store, "count", 0))})
        elif self.path.startswith("/status"):
            self._send(200, _status())
        elif self.path.startswith("/charts"):
            try:
                self._send(200, {"charts": _chart_specs()})
            except Exception as exc:
                self._send(500, {"error": str(exc)})
        elif self.path.startswith("/reports"):
            specs = _report_specs()
            self._send(200, {"reports": [
                {"id": i, "title": t, "sub": s}
                for i, (t, s, _f) in enumerate(specs)]})
        elif self.path.startswith("/scoreboard"):
            self._send(200, scoreboard.read())
        elif self.path.startswith("/rules"):
            self._send(200, {"path": _rules_path(), "text": _rules_text()})
        elif self.path.startswith("/warnings"):
            self._send(200, {"warnings": _STATE["warnings"][-50:]})
        elif self.path.startswith("/audit"):
            self._send(200, {"entries": audit.recent(50)})
        else:
            self._send(404, {"error": "unknown endpoint"})

    def do_POST(self):                                    # noqa: N802
        agent = _STATE["agent"]
        body = self._body()          # always drain the body first: replying
        if not self._authorized():   # before reading it desyncs keep-alive
            self._send(401, {"error": "missing or wrong X-Api-Key"})
            return
        if self.path.startswith("/ask"):
            q = (body.get("q") or "").strip()
            if not q:
                self._send(400, {"error": "missing 'q'"})
                return
            t0 = time.monotonic()
            with _STATE["lock"]:
                try:
                    res = agent.handle(q)
                    if isinstance(res, tuple) and res and isinstance(res[0], dict):
                        res = res[0]
                    if not isinstance(res, dict):
                        res = {"text": str(res)}
                except Exception as exc:
                    self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
                    return
            audit.log(q, res, time.monotonic() - t0, client="api")
            self._send(200, _json_safe(res))
        elif self.path.startswith("/report"):
            try:
                self._send(200, _json_safe(_run_report(int(body.get("id", -1)))))
            except Exception as exc:
                self._send(500, {"error": str(exc)})
        elif self.path.startswith("/rules"):
            text = body.get("text")
            if not isinstance(text, str):
                self._send(400, {"error": "missing 'text'"})
                return
            p = _rules_path()
            try:
                os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(text)
            except Exception as exc:
                self._send(500, {"error": str(exc)})
                return
            # house_rules re-reads on mtime change: effective immediately
            self._send(200, {"ok": True, "path": p})
        elif self.path.startswith("/clear_history"):
            try:
                agent.clear_history()
            except Exception:
                pass
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "unknown endpoint"})


_WEB_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Kronospan — Financial Data Assistant</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0b0f16;--panel:#111624;--panel2:#161d2e;--border:#222b40;
--text:#e8ecf4;--muted:#8f9ab0;--accent:#3d8bfd}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:14px 'Segoe UI',system-ui,sans-serif;display:flex;flex-direction:column;
min-height:100vh}
header{display:flex;gap:12px;align-items:center;padding:14px 22px;
border-bottom:1px solid var(--border)}
header b{font-size:17px}#stat{color:var(--muted);font-size:12px}
main{flex:1;max-width:980px;width:100%;margin:0 auto;padding:18px 22px 120px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:12px;
padding:14px 16px;margin:10px 0}
.q{color:var(--accent);font-weight:600}
table{border-collapse:collapse;margin-top:8px;font-size:13px;max-width:100%;
display:block;overflow-x:auto}
td,th{border:1px solid var(--border);padding:5px 9px;text-align:left}
th{background:var(--panel2);color:var(--muted)}
.muted{color:var(--muted);font-size:11px;margin-top:6px}
form{position:fixed;bottom:0;left:0;right:0;background:var(--panel);
border-top:1px solid var(--border);padding:12px 22px;display:flex;gap:8px}
input,select,button{background:var(--panel2);color:var(--text);
border:1px solid var(--border);border-radius:8px;padding:9px 12px;font:inherit}
input#q{flex:1}button{cursor:pointer}
button.primary{background:var(--accent);border:none;color:#fff;font-weight:700}
#key{width:130px}
</style></head><body>
<header><b>kronospan</b><span style="color:var(--muted)">Financial Data Assistant</span>
<span id="stat">connecting…</span>
<span style="flex:1"></span>
<select id="rep"><option value="">Pre-built reports…</option></select>
<button onclick="runReport()">Run</button>
</header>
<main id="log">
<div class="card">Ask about your data in plain language — answers come from the
offline model on this server. All data stays on this machine.
<div class="muted">Tip: name the table (LTL, CY01, WCR) and the date
("as at 31.07.2023") for the most precise answers.</div></div>
</main>
<form onsubmit="ask(event)">
<input id="key" placeholder="API key" type="password">
<input id="q" placeholder="e.g. total LTL outstanding per lender as at 31/12/2025" autofocus>
<button class="primary">Ask</button>
</form>
<script>
const log=document.getElementById('log'),stat=document.getElementById('stat');
const keyEl=document.getElementById('key');
keyEl.value=localStorage.getItem('apikey')||'';
keyEl.onchange=()=>localStorage.setItem('apikey',keyEl.value);
function hdrs(){const h={'Content-Type':'application/json'};
 if(keyEl.value)h['X-Api-Key']=keyEl.value;return h}
function card(html){const d=document.createElement('div');d.className='card';
 d.innerHTML=html;log.appendChild(d);d.scrollIntoView({behavior:'smooth'});return d}
async function boot(){try{
 const s=await(await fetch('/status',{headers:hdrs()})).json();
 stat.textContent=`${(s.tables||[]).length} tables · ${(s.docs||[]).length} PDFs · ${s.model||''}`;
 (s.warnings||[]).forEach(w=>card('<span style="color:#fbbf24">⚠ '+esc(w.message)+'</span>'));
 const sb=await(await fetch('/scoreboard',{headers:hdrs()})).json();
 if(sb.data_checks)stat.textContent+=` · verified ${sb.data_checks.passed}/${sb.data_checks.total}`;
 const r=await(await fetch('/reports',{headers:hdrs()})).json();
 (r.reports||[]).forEach(x=>{const o=document.createElement('option');
  o.value=x.id;o.textContent=x.title;document.getElementById('rep').appendChild(o)});
}catch(e){stat.textContent='backend unreachable'}}
function esc(s){return (s??'').toString().replace(/[&<>]/g,
 c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function ask(ev){ev.preventDefault();
 const q=document.getElementById('q').value.trim();if(!q)return;
 document.getElementById('q').value='';
 card('<span class="q">You</span><br>'+esc(q));
 const w=card('<span class="muted">thinking…</span>');
 try{const r=await fetch('/ask',{method:'POST',headers:hdrs(),
   body:JSON.stringify({q})});
  const res=await r.json();
  if(!r.ok){w.innerHTML='<span style="color:#f87171">'+esc(res.error||r.status)+'</span>';return}
  let h='';
  if(res.text)h+=esc(res.text).replace(/\n/g,'<br>');
  if(res.table_html)h+=res.table_html;
  const used=((res.code||'').match(/dfs\[\s*['"]([^'"]+)['"]\s*\]/g)||[])
    .map(m=>m.replace(/dfs\[\s*['"]|['"]\s*\]/g,''));
  if(used.length)h+='<div class="muted">Source tables: '+esc([...new Set(used)].join('; '))+'</div>';
  w.innerHTML=h||'<span class="muted">(no answer)</span>';
 }catch(e){w.innerHTML='<span style="color:#f87171">'+esc(e.message)+'</span>'}}
async function runReport(){const id=document.getElementById('rep').value;
 if(id==='')return;
 const w=card('<span class="muted">running report…</span>');
 try{const res=await(await fetch('/report',{method:'POST',headers:hdrs(),
   body:JSON.stringify({id:+id})})).json();
  let h='<span class="q">'+esc(res.title)+'</span><div class="muted">'+esc(res.sub)+'</div><table><tr>';
  (res.columns||[]).forEach(c=>h+='<th>'+esc(c)+'</th>');h+='</tr>';
  (res.rows||[]).slice(0,200).forEach(r=>{h+='<tr>';
   r.forEach(v=>h+='<td>'+esc(typeof v==='number'?v.toLocaleString():v)+'</td>');h+='</tr>'});
  h+='</table>';
  if((res.rows||[]).length>200)h+='<div class="muted">first 200 of '+res.rows.length+' rows</div>';
  w.innerHTML=h;
 }catch(e){w.innerHTML='<span style="color:#f87171">'+esc(e.message)+'</span>'}}
boot();
</script></body></html>"""


def main() -> None:
    host = os.environ.get("RAG_API_HOST", "0.0.0.0")
    port = int(os.environ.get("RAG_API_PORT", "8800"))
    _boot()
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"[server] listening on http://{host}:{port}  "
          f"(index: {cfg.INDEX_DIR}, ollama: {cfg.OLLAMA_HOST})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
