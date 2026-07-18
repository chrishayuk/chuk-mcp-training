//! The operator dashboard (spec §9): one page, four bands — Fleet · Runs ·
//! Money · Health — served by the control plane itself. Self-contained (no
//! external assets), dark ops theme, auto-refreshing, API token in
//! localStorage. Reads the same `/api/*` the MCP tools do; state-changing
//! actions (teardown) go through the bearer API.

use axum::response::Html;

const REFRESH_MS: u32 = 5000;

pub async fn page() -> Html<String> {
    Html(PAGE.replace("{REFRESH_MS}", &REFRESH_MS.to_string()))
}

const PAGE: &str = r####"<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>chuk-train</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{
  --plane:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --baseline:#383835; --ring:rgba(255,255,255,0.10);
  --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b; --accent:#3987e5;
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:14px;line-height:1.4}
a{color:var(--accent)}
header{display:flex;align-items:baseline;gap:1rem;flex-wrap:wrap;
  padding:1rem 1.5rem;border-bottom:1px solid var(--ring)}
header h1{font-size:1.05rem;margin:0;color:var(--ink)}
header .sub{color:var(--muted);font-size:.85rem}
header .spacer{flex:1}
header input{background:var(--surface);color:var(--ink);border:1px solid var(--ring);
  border-radius:6px;padding:.35rem .55rem;width:20rem;font-family:inherit}
header .dot{font-size:.8rem;color:var(--muted)}
main{padding:1.25rem 1.5rem;max-width:1200px;margin:0 auto}
section{margin-bottom:1.75rem}
h2{font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);
  margin:0 0 .6rem;font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.75rem}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:.85rem 1rem}
.tile .v{font-size:1.7rem;font-weight:600;color:var(--ink)}
.tile .v.sub{font-size:1rem;color:var(--ink2);font-weight:500}
.tile .l{font-size:.78rem;color:var(--muted);margin-top:.15rem}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;overflow:hidden}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{text-align:left;padding:.5rem .8rem;border-bottom:1px solid var(--grid);white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:.72rem;letter-spacing:.04em;text-transform:uppercase}
tr:last-child td{border-bottom:none}
td.num{font-variant-numeric:tabular-nums}
.st{display:inline-flex;align-items:center;gap:.4rem}
.st::before{content:"";width:.55rem;height:.55rem;border-radius:50%;background:var(--muted);flex:none}
.st.good::before{background:var(--good)} .st.run::before{background:var(--accent)}
.st.bad::before{background:var(--critical)} .st.warn::before{background:var(--warning)}
.st.mut::before{background:var(--muted)}
.btn{background:transparent;color:var(--critical);border:1px solid var(--critical);
  border-radius:6px;padding:.2rem .55rem;font-size:.78rem;cursor:pointer;font-family:inherit}
.btn:hover{background:var(--critical);color:#fff}
.spark{vertical-align:middle}
.empty{color:var(--muted);padding:.8rem;font-size:.85rem}
.err{color:var(--warning);font-size:.8rem;padding:.5rem .8rem}
.foot{color:var(--muted);font-size:.75rem;text-align:right;margin-top:.3rem}
</style></head><body>
<header>
  <h1>chuk-mcp-training</h1><span class="sub">operator dashboard</span>
  <span class="spacer"></span>
  <span class="dot" id="beat">—</span>
  <input id="tok" type="password" placeholder="API token" autocomplete="off">
</header>
<main>
  <section id="health"><h2>Health</h2><div class="tiles" id="healthTiles"></div></section>
  <section id="fleet"><h2>Fleet</h2><div class="card"><div id="fleetBody"></div></div></section>
  <section id="runs"><h2>Runs</h2><div class="card"><div id="runsBody"></div></div></section>
  <section id="money"><h2>Money</h2><div class="tiles" id="moneyTiles"></div>
    <div class="card" id="moneyCard" style="margin-top:.75rem"></div></section>
  <div class="foot" id="foot"></div>
</main>
<script>
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const $ = id => document.getElementById(id);
$("tok").value = localStorage.tok || "";
$("tok").addEventListener("change", e => { localStorage.tok = e.target.value; tick(); });

async function api(path, opts){
  const headers = {Authorization: "Bearer " + (localStorage.tok || "")};
  if (opts && opts.body) headers["Content-Type"] = "application/json";
  for (let attempt = 0; attempt < 2; attempt++){
    try{
      const r = await fetch(path, Object.assign({headers}, opts));
      if (r.ok) return r.status === 204 ? null : r.json();
      if (r.status < 500) throw new Error("http_" + r.status);
    }catch(e){ if (attempt) throw e; }
    await new Promise(res => setTimeout(res, 300));
  }
  throw new Error("request failed");
}

// -- status → class + human label -----------------------------------------
const RUN_ST = {completed:"good", running:"run", failed:"bad", cancelled:"bad",
                queued:"mut", assigned:"mut"};
const WK_ST  = {connected:"good", disconnected:"bad"};
const LEASE_ST = {active:"good", draining:"warn", destroyed:"mut"};
const pill = (cls, label) => `<span class="st ${cls}">${esc(label)}</span>`;

function ago(sec){ sec = Math.max(0, sec|0);
  if (sec < 60) return sec + "s"; if (sec < 3600) return (sec/60|0) + "m";
  return (sec/3600|0) + "h"; }

// -- sparkline: single-series loss, 2px accent line, recessive ------------
function spark(points){
  if (!points || points.length < 2) return `<span class="empty">—</span>`;
  const W = 120, H = 26, pad = 2;
  const xs = points.map(p => p.step), ys = points.map(p => p.value);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const sx = s => pad + (x1 === x0 ? 0 : (s - x0) / (x1 - x0)) * (W - 2*pad);
  const sy = v => pad + (y1 === y0 ? 0.5 : (v - y0) / (y1 - y0)) * (H - 2*pad); // loss: high=top? invert so lower=lower
  const d = points.map(p => `${sx(p.step).toFixed(1)},${(H - sy(p.value)).toFixed(1)}`).join(" ");
  const last = ys[ys.length-1];
  return `<svg class="spark" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" aria-label="loss ${last.toFixed(2)}">`
    + `<polyline fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="${d}"/>`
    + `</svg> <span class="num" style="color:var(--ink2)">${last.toFixed(2)}</span>`;
}

let inflight = false;
async function tick(){
  if (inflight) return; inflight = true;
  try{
    const [fleet, runs, spend] = await Promise.all([
      api("/api/fleet"), api("/api/runs?limit=25"), api("/api/spend"),
    ]);
    renderHealth(fleet, runs, spend);
    renderFleet(fleet);
    renderRuns(runs);
    renderMoney(spend);
    $("beat").textContent = "updated " + new Date().toLocaleTimeString();
    $("foot").textContent = "";
    // async enrich: loss sparklines + checkpoint age for train runs
    enrichRuns(runs);
  }catch(e){
    $("foot").textContent = "error: " + e.message + " (check API token)";
  }finally{ inflight = false; }
}

function renderHealth(fleet, runs, spend){
  const conn = fleet.filter(w => w.state === "connected").length;
  const running = runs.filter(r => r.state === "running").length;
  const queued = runs.filter(r => r.state === "queued").length;
  const leases = fleet.filter(w => w.lease && w.lease.state !== "destroyed").length;
  const draining = fleet.filter(w => w.lease && w.lease.state === "draining").length;
  const tiles = [
    ["Connected", conn, "workers"],
    ["Running", running, "runs"],
    ["Queued", queued, "runs"],
    ["Active leases", leases, draining ? draining + " draining" : "on providers"],
  ];
  $("healthTiles").innerHTML = tiles.map(([l,v,s]) =>
    `<div class="tile"><div class="v">${esc(v)}</div><div class="l">${esc(l)}${s?" · "+esc(s):""}</div></div>`).join("");
}

function renderFleet(fleet){
  if (!fleet.length){ $("fleetBody").innerHTML = `<div class="empty">no workers</div>`; return; }
  const rows = fleet.map(w => {
    const cls = WK_ST[w.state] || "mut";
    const gpu = (w.hardware && w.hardware.gpu) || "cpu";
    const vram = w.hardware && w.hardware.vram_mb ? " · " + Math.round(w.hardware.vram_mb/1024) + "GB" : "";
    let lease = `<span class="empty">—</span>`, action = "";
    if (w.lease){
      const L = w.lease, remain = Math.max(0, L.granted_min + (L.extensions||[]).reduce((a,e)=>a+e.minutes,0)
        - (Date.now()/1000 - L.started_at)/60);
      lease = `${esc(L.provider)} · $${L.price_hr.toFixed(2)}/h · ${pill(LEASE_ST[L.state]||"mut", L.state)}`
        + ` · <span class="num">${remain.toFixed(1)}m left</span>`;
      if (L.state !== "destroyed")
        action = `<button class="btn" onclick="teardown('${esc(w.id)}')">teardown</button>`;
    }
    return `<tr><td>${esc(w.id)}</td><td>${esc(gpu)}${vram}</td><td>${pill(cls, w.state)}</td>`
      + `<td class="num">${ago(w.heartbeat_age_s)}</td><td>${esc(w.current_run||"—")}</td>`
      + `<td>${lease}</td><td>${action}</td></tr>`;
  }).join("");
  $("fleetBody").innerHTML = `<table><tr><th>worker</th><th>gpu</th><th>state</th><th>hb</th>`
    + `<th>run</th><th>lease</th><th></th></tr>${rows}</table>`;
}

function renderRuns(runs){
  if (!runs.length){ $("runsBody").innerHTML = `<div class="empty">no runs</div>`; return; }
  const rows = runs.map(r => {
    const cls = RUN_ST[r.state] || "mut";
    const exit = r.exit_code === null || r.exit_code === undefined ? "" : r.exit_code;
    return `<tr data-run="${esc(r.id)}"><td>${esc(r.id)}</td><td>${esc(r.name)}</td>`
      + `<td>${esc(r.kind)}</td><td>${pill(cls, r.state)}</td><td>${esc(r.worker_id||"—")}</td>`
      + `<td class="loss">${r.kind === "train" ? '<span class="empty">…</span>' : ""}</td>`
      + `<td class="ckpt num"></td><td class="num">${exit}</td></tr>`;
  }).join("");
  $("runsBody").innerHTML = `<table><tr><th>id</th><th>name</th><th>kind</th><th>state</th>`
    + `<th>worker</th><th>loss</th><th>ckpt</th><th>exit</th></tr>${rows}</table>`;
}

async function enrichRuns(runs){
  const train = runs.filter(r => r.kind === "train").slice(0, 12);
  await Promise.all(train.map(async r => {
    const row = document.querySelector(`tr[data-run="${CSS.escape(r.id)}"]`);
    if (!row) return;
    try{
      const [m, c] = await Promise.all([
        api(`/api/runs/${r.id}/metrics?keys=loss&downsample=60`),
        api(`/api/runs/${r.id}/checkpoints`),
      ]);
      const pts = (m.series && m.series.loss) || [];
      row.querySelector(".loss").innerHTML = spark(pts);
      const last = c.length ? c[c.length-1] : null;
      row.querySelector(".ckpt").textContent = last ? "step " + last.step : "—";
    }catch(e){ /* leave placeholder */ }
  }));
}

function renderMoney(spend){
  $("moneyTiles").innerHTML = [
    ["Spent", "$" + (spend.total_spent||0).toFixed(4), "realised (ledger)"],
    ["Committed", "$" + (spend.total_committed||0).toFixed(4), "live leases"],
  ].map(([l,v,s]) => `<div class="tile"><div class="v sub">${esc(v)}</div><div class="l">${esc(l)} · ${esc(s)}</div></div>`).join("");
  if (!spend.lines || !spend.lines.length){ $("moneyCard").innerHTML = `<div class="empty">no spend yet</div>`; return; }
  const rows = spend.lines.map(l => `<tr><td>${esc(l.provider)}</td>`
    + `<td class="num">$${l.committed.toFixed(4)}</td><td class="num">$${l.spent.toFixed(4)}</td></tr>`).join("");
  $("moneyCard").innerHTML = `<table><tr><th>provider</th><th>committed</th><th>spent</th></tr>${rows}</table>`;
}

async function teardown(id){
  if (!confirm(`Tear down ${id}? Drains, then destroys the instance (provider-verified).`)) return;
  try{ await api(`/api/workers/${id}/teardown`, {method:"POST", body: JSON.stringify({force:false})}); tick(); }
  catch(e){ alert("teardown failed: " + e.message); }
}

setInterval(tick, {REFRESH_MS}); tick();
</script></body></html>"####;
