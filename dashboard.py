#!/usr/bin/env python3
"""
Fraud Sentinel — local dashboard generator.

Reads the EXISTING pipeline outputs (read-only) and builds a single, self-contained
dashboard.html you can open in any browser. No server, no Ollama, no retraining, and
no modification of the core fraud logic or its output.

Why static HTML instead of Streamlit: Streamlit fails to import in this project's
Python env (a protobuf descriptor incompatibility). A self-contained HTML file has
zero runtime dependencies and is the most reliable way to ship the dashboard.

Inputs (paths relative to this file):
  fraud_output.jsonl        (required)  verdicts + justifications
  transactions.csv          (optional)  transaction detail fields
  accounts.csv              (optional)  account detail fields
  customers.csv             (optional)  customer detail fields
  data_quality_report.json  (optional)  verified data-quality metrics

Output:
  dashboard.html

Usage:
  python3 dashboard.py            # writes dashboard.html
  open dashboard.html             # (macOS) view it
"""
import csv, html, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---- robust loaders (handle missing files / malformed lines / missing cols) ----
def load_jsonl(path):
    rows, bad = [], 0
    if not path.exists():
        return rows, bad, f"missing file: {path.name}"
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                bad += 1
    return rows, bad, None

def load_csv_index(path, key):
    """Return {key_value: row_dict}. Empty on any failure."""
    idx = {}
    if not path.exists():
        return idx
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                k = r.get(key)
                if k:
                    idx[k] = r
    except Exception:
        pass
    return idx

def g(d, k):
    v = d.get(k) if isinstance(d, dict) else None
    return "" if v is None else str(v)

def main():
    out_rows, bad_lines, err = load_jsonl(HERE / "fraud_output.jsonl")

    txn_idx  = load_csv_index(HERE / "transactions.csv", "transaction_id")
    acct_idx = load_csv_index(HERE / "accounts.csv", "account_id")
    cust_idx = load_csv_index(HERE / "customers.csv", "customer_id")

    dq = None
    dq_path = HERE / "data_quality_report.json"
    if dq_path.exists():
        try:
            dq = json.load(open(dq_path))
        except Exception:
            dq = None

    # ---- build per-transaction records (verdict is taken AS-IS, never recomputed) ----
    records = []
    for r in out_rows:
        tid = str(r.get("transaction_id", ""))
        t = txn_idx.get(tid, {})
        acc_id = g(t, "account_id")
        a = acct_idx.get(acc_id, {})
        # account.customer_id is the source of truth; fall back to txn's
        cust_id = g(a, "customer_id") or g(t, "customer_id")
        c = cust_idx.get(cust_id, {})
        try:
            conf = float(r.get("confidence"))
        except Exception:
            conf = None
        records.append({
            "transaction_id": tid,
            "is_fraud": bool(r.get("is_fraud")),
            "confidence": conf,
            "justification": str(r.get("justification", "")),
            "amount": g(t, "amount"),
            "currency": g(t, "currency"),
            "timestamp": g(t, "transaction_timestamp"),
            "type": g(t, "transaction_type"),
            "channel": g(t, "channel"),
            "status": g(t, "status"),
            "merchant_name": g(t, "merchant_name"),
            "merchant_category": g(t, "merchant_category"),
            "merchant_city": g(t, "merchant_city"),
            "is_foreign": g(t, "is_foreign_transaction"),
            "is_new_device": g(t, "is_new_device"),
            "device_type": g(t, "device_type"),
            "distance_km": g(t, "distance_from_home_km"),
            "amt_ratio": g(t, "amount_to_account_avg_ratio"),
            "txn_24h": g(t, "txn_count_last_24h"),
            "account_id": acc_id,
            "account_type": g(a, "account_type"),
            "account_status": g(a, "account_status"),
            "current_balance": g(a, "current_balance"),
            "account_tier": g(a, "account_tier"),
            "customer_id": cust_id,
            "risk_rating": g(c, "risk_rating"),
            "kyc_status": g(c, "kyc_status"),
            "customer_segment": g(c, "customer_segment"),
            "is_pep": g(c, "is_politically_exposed"),
            "cust_city": g(c, "city"),
            "cust_state": g(c, "state"),
        })

    # ---- dynamic metrics (never hardcoded) ----
    total = len(records)
    flagged = sum(1 for r in records if r["is_fraud"])
    legit = total - flagged
    pct = (flagged / total * 100) if total else 0.0
    confs = [r["confidence"] for r in records if r["confidence"] is not None]
    conf_avg = round(sum(confs) / len(confs), 3) if confs else 0.0
    metrics = {
        "total": total, "flagged": flagged, "legit": legit,
        "pct": round(pct, 1), "conf_avg": conf_avg,
        "bad_lines": bad_lines,
        "have_txn": bool(txn_idx), "have_acct": bool(acct_idx), "have_cust": bool(cust_idx),
    }

    payload = json.dumps({"records": records, "metrics": metrics, "dq": dq})
    html_doc = HTML_TEMPLATE.replace("__PAYLOAD__", payload).replace(
        "__ERROR__", html.escape(err) if err else "")

    out = HERE / "dashboard.html"
    out.write_text(html_doc)
    print(f"wrote {out}")
    print(f"  transactions: {total} | flagged: {flagged} ({metrics['pct']}%) | "
          f"legit: {legit} | malformed lines skipped: {bad_lines}")
    if err:
        print(f"  WARNING: {err}")
    print(f"\nOpen it with:\n  open dashboard.html      # macOS\n"
          f"  python3 -m http.server 8501   # then visit http://localhost:8501/dashboard.html")

# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fraud Sentinel Dashboard</title>
<style>
:root{--bg:#0f1420;--panel:#1a2233;--line:#2a3552;--txt:#e6ebf5;--mut:#9aa7c2;
--accent:#4f8cff;--danger:#ff5c7a;--ok:#35c28f;--warn:#f0b429;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{padding:20px 24px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{margin:0;font-size:20px}
.sub{color:var(--mut);font-size:13px;margin-top:4px}
.banner{background:#3a2a10;border:1px solid var(--warn);color:#ffe9b8;
padding:10px 14px;margin:16px 24px;border-radius:8px;font-size:13px}
.wrap{padding:0 24px 40px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:18px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
.card .n{font-size:26px;font-weight:700}
.card .l{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.danger{color:var(--danger)}.ok{color:var(--ok)}.accent{color:var(--accent)}
.grid{display:grid;grid-template-columns:1.35fr .9fr;gap:18px;align-items:start}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
.panel h2{margin:0 0 12px;font-size:15px}
.controls{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:12px}
input,select{background:var(--bg);color:var(--txt);border:1px solid var(--line);
border-radius:8px;padding:8px 10px;font-size:13px}
input[type=text]{min-width:200px;flex:1}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mut);font-weight:600;cursor:pointer;user-select:none;position:sticky;top:0;background:var(--panel)}
tbody tr{cursor:pointer}
tbody tr:hover{background:#20293d}
tbody tr.sel{background:#243250}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700}
.tag.f{background:rgba(255,92,122,.15);color:var(--danger)}
.tag.l{background:rgba(53,194,143,.15);color:var(--ok)}
.tblwrap{max-height:520px;overflow:auto;border:1px solid var(--line);border-radius:8px}
.kv{display:grid;grid-template-columns:130px 1fr;gap:4px 12px;font-size:13px}
.kv .k{color:var(--mut)}
.kv .v{word-break:break-word;white-space:normal}
.sect{margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}
.sect h3{margin:0 0 8px;font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}
.just{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px;margin-top:6px;white-space:normal}
.muted{color:var(--mut)}
.dq{font-size:13px}
.dq .row{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--line)}
.dq .row span:last-child{font-weight:600}
.err{background:rgba(255,92,122,.12);border:1px solid var(--danger);color:#ffc2ce;
padding:12px;border-radius:8px;margin:16px 0}
.footer{color:var(--mut);font-size:12px;margin-top:24px}
</style></head>
<body>
<header>
  <h1>Relational Data Wrangler &amp; Fraud Sentinel</h1>
  <div class="sub">Local dashboard · read-only view of the existing pipeline output</div>
</header>
<div class="banner">
  <b>Important:</b> Risk flags are <b>not confirmed fraud</b> — they are heuristic + anomaly
  risk scores. <b>Confidence is a boundary-distance signal, not a calibrated probability.</b>
  No ground-truth fraud labels exist in this dataset.
</div>
<div class="wrap">
  <div id="err"></div>
  <div class="cards" id="cards"></div>
  <div class="grid">
    <div class="panel">
      <h2>Transaction explorer</h2>
      <div class="controls">
        <input type="text" id="q" placeholder="Search transaction ID…">
        <select id="verdict">
          <option value="all">All verdicts</option>
          <option value="fraud">Flagged (risk)</option>
          <option value="legit">Not flagged</option>
        </select>
        <select id="minconf">
          <option value="0">Confidence ≥ 0.0</option>
          <option value="0.5">Confidence ≥ 0.5</option>
          <option value="0.7">Confidence ≥ 0.7</option>
          <option value="0.8">Confidence ≥ 0.8</option>
          <option value="0.9">Confidence ≥ 0.9</option>
        </select>
      </div>
      <div class="tblwrap">
        <table id="tbl">
          <thead><tr>
            <th data-k="transaction_id">Txn ID</th>
            <th data-k="is_fraud">Verdict</th>
            <th data-k="confidence">Conf</th>
            <th data-k="amount">Amount</th>
            <th data-k="merchant_name">Merchant</th>
          </tr></thead>
          <tbody id="tbody"></tbody>
        </table>
      </div>
      <div class="muted" id="count" style="margin-top:8px"></div>
    </div>
    <div>
      <div class="panel" id="detail"><h2>Transaction details</h2>
        <div class="muted">Select a row to see details.</div></div>
      <div class="panel" id="dqpanel" style="margin-top:18px"><h2>Data quality</h2></div>
    </div>
  </div>
  <div class="footer">Generated by dashboard.py from local files. Verdicts and confidence
  values are shown exactly as produced by fraud_sentinel.py; this view does not recompute them.</div>
</div>
<script>
const DATA = __PAYLOAD__;
const SERVER_ERR = "__ERROR__";
const $ = s => document.querySelector(s);
const esc = s => (s==null?"":String(s)).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const recs = DATA.records||[], M = DATA.metrics||{}, DQ = DATA.dq;
let sortK="confidence", sortDir=-1, selId=null;

if (SERVER_ERR){ $("#err").innerHTML='<div class="err"><b>Cannot load output:</b> '+esc(SERVER_ERR)+'</div>'; }

function cards(){
  const c=[
    ["Total transactions", M.total, ""],
    ["Flagged (risk)", M.flagged, "danger"],
    ["Not flagged", M.legit, "ok"],
    ["Flag rate", (M.pct!=null?M.pct+"%":"—"), "accent"],
    ["Avg confidence", (M.conf_avg!=null?M.conf_avg:"—"), ""],
  ];
  $("#cards").innerHTML=c.map(x=>`<div class="card"><div class="n ${x[2]}">${esc(x[1])}</div><div class="l">${esc(x[0])}</div></div>`).join("");
}
function fmtConf(v){return v==null?"—":Number(v).toFixed(2);}
function fmtAmt(r){return r.amount?(esc(r.amount)+(r.currency?" "+esc(r.currency):"")):"—";}

function filtered(){
  const q=$("#q").value.trim().toLowerCase(), vd=$("#verdict").value, mc=parseFloat($("#minconf").value);
  let rows=recs.filter(r=>{
    if(q && !r.transaction_id.toLowerCase().includes(q)) return false;
    if(vd==="fraud" && !r.is_fraud) return false;
    if(vd==="legit" && r.is_fraud) return false;
    if(r.confidence!=null && r.confidence<mc) return false;
    if(r.confidence==null && mc>0) return false;
    return true;
  });
  rows.sort((a,b)=>{
    let x=a[sortK], y=b[sortK];
    if(sortK==="confidence"){x=x==null?-1:x; y=y==null?-1:y;}
    if(sortK==="amount"){x=parseFloat(x)||0; y=parseFloat(y)||0;}
    if(typeof x==="string"){return sortDir*x.localeCompare(y);}
    return sortDir*(x<y?-1:x>y?1:0);
  });
  return rows;
}
function renderTable(){
  const rows=filtered();
  $("#tbody").innerHTML=rows.map(r=>`<tr data-id="${esc(r.transaction_id)}" class="${r.transaction_id===selId?'sel':''}">
    <td>${esc(r.transaction_id)}</td>
    <td><span class="tag ${r.is_fraud?'f':'l'}">${r.is_fraud?'FLAGGED':'clear'}</span></td>
    <td>${fmtConf(r.confidence)}</td>
    <td>${fmtAmt(r)}</td>
    <td>${esc(r.merchant_name)||'—'}</td></tr>`).join("");
  $("#count").textContent=rows.length+" of "+recs.length+" transactions";
  document.querySelectorAll("#tbody tr").forEach(tr=>tr.onclick=()=>{selId=tr.dataset.id;renderTable();detail();});
}
function kv(pairs){return '<div class="kv">'+pairs.map(p=>`<div class="k">${esc(p[0])}</div><div class="v">${esc(p[1])||'—'}</div>`).join("")+'</div>';}
function detail(){
  const r=recs.find(x=>x.transaction_id===selId);
  if(!r){return;}
  $("#detail").innerHTML=`<h2>Transaction details</h2>
    <div style="margin-bottom:6px"><span class="tag ${r.is_fraud?'f':'l'}">${r.is_fraud?'FLAGGED (risk)':'not flagged'}</span>
    &nbsp;confidence <b>${fmtConf(r.confidence)}</b></div>
    <div class="just">${esc(r.justification)||'<span class="muted">no justification</span>'}</div>
    <div class="sect"><h3>Transaction</h3>${kv([
      ["Txn ID",r.transaction_id],["Timestamp",r.timestamp],["Amount",fmtAmtPlain(r)],
      ["Type",r.type],["Channel",r.channel],["Status",r.status],
      ["Merchant",r.merchant_name],["Category",r.merchant_category],["Merchant city",r.merchant_city],
      ["Foreign?",r.is_foreign],["New device?",r.is_new_device],["Device",r.device_type],
      ["Distance km",r.distance_km],["Amt/avg ratio",r.amt_ratio],["Txns 24h",r.txn_24h],
    ])}</div>
    <div class="sect"><h3>Account</h3>${DATA.metrics.have_acct?kv([
      ["Account ID",r.account_id],["Type",r.account_type],["Status",r.account_status],
      ["Balance",r.current_balance],["Tier",r.account_tier],
    ]):'<span class="muted">accounts.csv not available</span>'}</div>
    <div class="sect"><h3>Customer</h3>${DATA.metrics.have_cust?kv([
      ["Customer ID",r.customer_id],["Risk rating",r.risk_rating],["KYC",r.kyc_status],
      ["Segment",r.customer_segment],["PEP?",r.is_pep],["Location",(r.cust_city+(r.cust_state?", "+r.cust_state:""))],
    ]):'<span class="muted">customers.csv not available</span>'}</div>`;
}
function fmtAmtPlain(r){return r.amount?(r.amount+(r.currency?" "+r.currency:"")):"";}

function dqPanel(){
  const el=$("#dqpanel");
  if(!DQ){el.innerHTML='<h2>Data quality</h2><div class="muted">data_quality_report.json not found. Run <code>python3 data_quality.py</code> to generate it.</div>';return;}
  const rc=DQ.record_counts||{}, dup=DQ.duplicates_removed||{}, inv=DQ.invalid_or_corrupt||{}, orp=DQ.orphaned_relationships||{}, inj=DQ.prompt_injection||{};
  const row=(k,v)=>`<div class="row"><span>${esc(k)}</span><span>${esc(v)}</span></div>`;
  el.innerHTML='<h2>Data quality</h2><div class="dq">'+
    row("Transactions (raw → final)", (rc.transactions_raw+" → "+rc.transactions_final))+
    row("Duplicate rows removed", dup.txn_full_duplicate_rows)+
    row("Amount: non-numeric", inv.amount_non_numeric)+
    row("Amount: non-positive", inv.amount_non_positive)+
    row("Timestamp: unparseable", inv.timestamp_unparseable)+
    row("Orphan txn → account", orp.txn_account_id_not_in_accounts)+
    row("Orphan txn → customer", orp.txn_customer_id_not_in_customers)+
    row("Injection strings flagged", inj.merchant_strings_flagged)+
    '</div><div class="muted" style="margin-top:8px">Metrics from the pipeline\'s own data_quality_report.json.</div>';
}
cards(); dqPanel();
["q","verdict","minconf"].forEach(id=>$("#"+id).addEventListener("input",renderTable));
document.querySelectorAll("#tbl th").forEach(th=>th.onclick=()=>{
  const k=th.dataset.k; if(k===sortK){sortDir*=-1;}else{sortK=k;sortDir=(k==="transaction_id"||k==="merchant_name")?1:-1;}
  renderTable();
});
renderTable();
</script>
</body></html>"""

if __name__ == "__main__":
    main()
