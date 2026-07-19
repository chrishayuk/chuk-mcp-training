//! The operator dashboard (spec §9): served by the control plane itself,
//! self-contained (no external assets), dark mission-control theme. Two views
//! behind a hash router — an **overview** (health · fleet · runs · money) and a
//! full **per-run** view (live loss curve, streamed logs, config, checkpoints
//! with metadata + download links, events, out-links). Reads the same `/api/*`
//! the MCP tools do; the Google session cookie (or the API-token box in local
//! dev) authenticates every call; live updates via polling.

use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{Html, IntoResponse, Redirect, Response};

use crate::AppState;

/// Overview refresh cadence; the per-run view polls faster (see RUN_MS in JS).
const REFRESH_MS: u32 = 4000;

/// Serve the dashboard. When Google sign-in is configured, a valid session
/// cookie is required (else redirect to login); otherwise the API-token box is
/// shown (local dev).
pub async fn dashboard(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if state.config.auth_enabled() {
        match crate::auth::session_email(&state, &headers) {
            Some(email) => Html(render(Some(&email), true)).into_response(),
            None => Redirect::to("/auth/login").into_response(),
        }
    } else {
        Html(render(None, false)).into_response()
    }
}

fn render(user: Option<&str>, auth_enabled: bool) -> String {
    let header_right = if auth_enabled {
        format!(
            r#"<span class="who">{} · <a href="/auth/logout">sign out</a></span>"#,
            user.unwrap_or("")
        )
    } else {
        r#"<input id="tok" type="password" placeholder="API token" autocomplete="off">"#.to_owned()
    };
    PAGE.replace("{REFRESH_MS}", &REFRESH_MS.to_string())
        .replace("{HEADER_RIGHT}", &header_right)
}

const PAGE: &str = r####"<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>chuk-train · operator dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{
  --plane:#0d0d0d; --surface:#1a1a19; --surface2:#212120; --ink:#f4f3ee; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --baseline:#3a3a37; --ring:rgba(255,255,255,.10);
  --ring2:rgba(255,255,255,.16);
  --good:#22b45a; --warning:#fab219; --serious:#ec835a; --critical:#e0483d;
  --accent:#3987e5; --accent2:#2bb1a8;
  --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--plane);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.45;-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
.mono,.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
button{font-family:inherit} :focus-visible{outline:2px solid var(--accent);outline-offset:2px}
header{display:flex;align-items:center;gap:.9rem;padding:.85rem 1.4rem;border-bottom:1px solid var(--ring);
  position:sticky;top:0;background:rgba(13,13,13,.86);backdrop-filter:blur(8px);z-index:20}
header .brand{font-weight:700;letter-spacing:.02em} header .brand b{color:var(--accent)}
header .tag{color:var(--muted);font-size:.82rem} header .spacer{flex:1}
header .live{display:inline-flex;align-items:center;gap:.4rem;color:var(--muted);font-size:.78rem}
header .live .dot{width:.5rem;height:.5rem;border-radius:50%;background:var(--good);animation:pulse 2.4s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(34,180,90,.45)}70%{box-shadow:0 0 0 6px rgba(34,180,90,0)}100%{box-shadow:0 0 0 0 rgba(34,180,90,0)}}
header .who{color:var(--muted);font-size:.8rem} header .who a{color:var(--muted)}
header input{background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:6px;padding:.35rem .55rem;width:18rem;font-family:inherit}
@media (prefers-reduced-motion:reduce){header .live .dot{animation:none}}
main{max-width:1220px;margin:0 auto;padding:1.3rem 1.4rem 3rem}
.eyebrow{font-size:.7rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);font-weight:600;margin:0 0 .55rem}
section{margin-bottom:1.7rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.7rem}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:.8rem .95rem}
.tile .v{font-size:1.6rem;font-weight:650;font-family:var(--mono);font-variant-numeric:tabular-nums;line-height:1.1}
.tile .v.s{font-size:1.15rem} .tile .l{font-size:.76rem;color:var(--muted);margin-top:.25rem} .tile .l b{color:var(--ink2);font-weight:600}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;overflow:hidden}
.card .hd{display:flex;align-items:center;gap:.6rem;padding:.6rem .9rem;border-bottom:1px solid var(--grid)}
.card .hd h3{margin:0;font-size:.74rem;letter-spacing:.07em;text-transform:uppercase;color:var(--ink2);font-weight:600}
.card .hd .sp{flex:1} .card .hd .tag{color:var(--muted);font-size:.74rem}
.tblwrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{text-align:left;padding:.52rem .9rem;border-bottom:1px solid var(--grid);white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:.68rem;letter-spacing:.05em;text-transform:uppercase}
tr:last-child td{border-bottom:none} tr.click{cursor:pointer} tr.click:hover{background:var(--surface2)}
td .name{font-weight:600} .empty{color:var(--muted);padding:.9rem;font-size:.86rem}
.st{display:inline-flex;align-items:center;gap:.42rem;font-size:.8rem;white-space:nowrap}
.st::before{content:"";width:.5rem;height:.5rem;border-radius:50%;background:var(--muted);flex:none}
.st.good::before{background:var(--good)} .st.run::before{background:var(--accent);animation:pulse2 2s infinite}
.st.bad::before{background:var(--critical)} .st.warn::before{background:var(--warning)} .st.mut::before{background:var(--muted)}
@keyframes pulse2{0%,100%{opacity:1}50%{opacity:.35}}
@media (prefers-reduced-motion:reduce){.st.run::before{animation:none}}
.chip{display:inline-flex;align-items:center;gap:.35rem;font-family:var(--mono);font-size:.74rem;background:var(--surface2);
  border:1px solid var(--ring);border-radius:999px;padding:.12rem .5rem;color:var(--ink2)}
.chip.hot{color:#e6b8a8;border-color:rgba(236,131,90,.4)} .chip.final{color:#a9d3f5;border-color:rgba(57,135,229,.4)}
.chip.drive{color:#a7e0c2;border-color:rgba(34,180,90,.4)} .pin{color:var(--warning)}
.btn{background:transparent;color:var(--critical);border:1px solid var(--critical);border-radius:6px;padding:.18rem .5rem;font-size:.76rem;cursor:pointer}
.btn.go{color:var(--accent);border-color:var(--accent)}
.runacts{display:flex;gap:.5rem;align-items:flex-start}
.btn:hover{background:var(--critical);color:#fff}
.runhead{display:flex;align-items:flex-start;gap:1rem;flex-wrap:wrap;margin-bottom:1.1rem}
.back{color:var(--muted);border:1px solid var(--ring);border-radius:8px;padding:.3rem .6rem;font-size:.8rem;cursor:pointer;background:transparent}
.back:hover{color:var(--ink);border-color:var(--ring2)}
.runid{font-family:var(--mono);font-size:1.05rem;color:var(--ink);font-weight:600;word-break:break-all}
.runsub{color:var(--muted);font-size:.84rem;margin-top:.15rem;display:flex;gap:.6rem;flex-wrap:wrap;align-items:center}
.runsub .name{color:var(--ink);font-weight:600}
.links{display:flex;gap:.5rem;flex-wrap:wrap}
.olink{display:inline-flex;align-items:center;gap:.4rem;font-size:.8rem;background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:.34rem .6rem;color:var(--ink2)}
.olink:hover{border-color:var(--ring2);color:var(--ink);text-decoration:none}
.olink .g{width:.5rem;height:.5rem;border-radius:2px;background:var(--accent)}
.olink.wandb .g{background:var(--warning)} .olink.exp .g{background:var(--accent2)} .olink.r2 .g{background:var(--serious)}
.telem{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:.55rem;margin-bottom:1.1rem}
.telem .t{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:.55rem .7rem}
.telem .t .k{font-size:.68rem;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.telem .t .val{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:1.05rem;margin-top:.15rem;color:var(--ink)}
.telem .t .val small{color:var(--muted);font-size:.72rem}
.sys{display:flex;flex-direction:column;gap:.55rem}
.sysrow{display:grid;grid-template-columns:3rem 1fr;align-items:center;gap:.6rem}
.sysrow .sk{font-size:.7rem;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
.sysrow .sv{display:flex;align-items:center;gap:.5rem;font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:.85rem;color:var(--ink)}
.sysrow .sv small{color:var(--muted);font-size:.72rem}
.sysrow .bar{flex:1;height:7px;background:var(--ring);border-radius:4px;overflow:hidden}
.sysrow .bar .fill{height:100%;background:var(--accent);border-radius:4px;transition:width .4s ease}
.sysrow .pv{min-width:2.6rem;text-align:right}
.grid2{display:grid;grid-template-columns:1.35fr 1fr;gap:1rem;align-items:start}
@media (max-width:920px){.grid2{grid-template-columns:1fr}}
.stack{display:flex;flex-direction:column;gap:1rem}
.bar{height:.5rem;background:var(--surface2);border-radius:999px;overflow:hidden;border:1px solid var(--ring)}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),#6aa8ee)}
.chartcard .body{padding:.7rem .5rem .4rem}
.metricsel{display:flex;gap:.4rem;flex-wrap:wrap}
.metricsel button{background:var(--surface2);border:1px solid var(--ring);color:var(--ink2);border-radius:7px;padding:.16rem .5rem;font-size:.74rem;cursor:pointer;font-family:var(--mono)}
.metricsel button[aria-pressed=true]{border-color:var(--accent);color:var(--ink)}
.chart{width:100%;height:210px;display:block}
.axis{fill:var(--muted);font-family:var(--mono);font-size:10px}
.logs{background:#0a0a0a;border-top:1px solid var(--grid);font-family:var(--mono);font-size:12.5px;line-height:1.55;height:300px;overflow:auto;padding:.6rem .9rem}
.logs .ln{white-space:pre-wrap;word-break:break-word;color:var(--ink2)} .logs .ln .t{color:var(--muted)}
.logs .ln.step{color:var(--ink)} .logs .ln.ck{color:var(--accent2)} .logs .ln.warnln{color:var(--warning)}
.logs::-webkit-scrollbar{width:10px}.logs::-webkit-scrollbar-thumb{background:#2a2a28;border-radius:6px}
.ckmeta td{background:var(--surface2);padding:.5rem .9rem .7rem 1.6rem;border-bottom:1px solid var(--grid)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:.15rem .9rem;max-width:680px;font-family:var(--mono);font-size:.76rem}
.kv dt{color:var(--muted)} .kv dd{margin:0;color:var(--ink2);word-break:break-all}
.expand{cursor:pointer;color:var(--muted);user-select:none} .expand:hover{color:var(--ink)}
.dl{color:var(--accent);font-size:.78rem;cursor:pointer}
.tl{padding:.4rem .9rem .7rem;max-height:300px;overflow:auto} .tl .ev{display:grid;grid-template-columns:auto 1fr auto;gap:.7rem;align-items:baseline;padding:.28rem 0}
#cks{max-height:360px;overflow:auto}
.filters{display:flex;gap:.4rem;flex-wrap:wrap;align-items:center}
.filters button{background:var(--surface2);border:1px solid var(--ring);color:var(--muted);border-radius:7px;padding:.14rem .5rem;font-size:.72rem;cursor:pointer;font-family:var(--mono)}
.filters button[aria-pressed=true]{border-color:var(--accent);color:var(--ink)}
.more{display:block;width:100%;background:var(--surface2);border:none;border-top:1px solid var(--grid);color:var(--ink2);padding:.5rem;font-size:.8rem;cursor:pointer}
.more:hover{color:var(--ink);background:var(--surface)}
.tl .ev .mk{width:.5rem;height:.5rem;border-radius:50%;background:var(--muted);margin-top:.35rem}
.tl .ev.run .mk{background:var(--accent)} .tl .ev.good .mk{background:var(--good)} .tl .ev.ck .mk{background:var(--accent2)} .tl .ev.bad .mk{background:var(--critical)}
.tl .ev .lbl{color:var(--ink2)} .tl .ev .lbl b{color:var(--ink);font-weight:600} .tl .ev .ts{color:var(--muted);font-family:var(--mono);font-size:.74rem}
.deflist{display:grid;grid-template-columns:auto 1fr;gap:.35rem .9rem;padding:.7rem .9rem;font-size:.85rem}
.deflist dt{color:var(--muted)} .deflist dd{margin:0;color:var(--ink2);font-family:var(--mono);word-break:break-all}
.foot{color:var(--muted);font-size:.74rem;text-align:right;margin-top:1rem}
.hidden{display:none} .err{color:var(--warning);font-size:.82rem;padding:.4rem 0}
.nav{color:var(--muted);font-size:.82rem;border-radius:6px;padding:.2rem .5rem}
.nav:hover{color:var(--ink);text-decoration:none}
.form{display:flex;gap:.5rem;flex-wrap:wrap;padding:.7rem .9rem;border-top:1px solid var(--grid);align-items:center}
.form input,.form select{background:var(--surface2);border:1px solid var(--ring);color:var(--ink);border-radius:7px;padding:.34rem .5rem;font-family:inherit;font-size:.85rem}
.form button{background:var(--accent);border:none;color:#fff;border-radius:7px;padding:.36rem .8rem;font-size:.82rem;cursor:pointer}
.form button:hover{filter:brightness(1.08)}
.rvk{color:var(--critical);cursor:pointer;font-size:.78rem}.rvk:hover{text-decoration:underline}
.rolechip{font-family:var(--mono);font-size:.72rem;padding:.08rem .45rem;border-radius:999px;border:1px solid var(--ring);color:var(--ink2)}
.keyreveal{margin:.7rem .9rem;padding:.7rem .9rem;background:#0a1a0f;border:1px solid rgba(34,180,90,.4);border-radius:8px}
.keyreveal .k{font-family:var(--mono);font-size:.82rem;color:var(--good);word-break:break-all;user-select:all}
.keyreveal .warn{color:var(--warning);font-size:.76rem;margin-top:.35rem}
</style></head><body>
<header>
  <span class="brand">chuk-<b>train</b></span><span class="tag">operator dashboard</span>
  <a class="nav" href="#/">overview</a>
  <a class="nav hidden" id="navAccess" href="#/access">access</a>
  <span class="spacer"></span>
  <span class="live"><span class="dot"></span><span id="clock">connecting…</span></span>
  {HEADER_RIGHT}
</header>
<main id="app"></main>
<script>
"use strict";
const OVERVIEW_MS={REFRESH_MS}, RUN_MS=2000;
const $=(s,r=document)=>r.querySelector(s);
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt=(n,d=2)=>(n===null||n===undefined||Number.isNaN(n))?"—":Number(n).toFixed(d);
const commas=n=>Number(n).toLocaleString("en-US");
const nows=()=>Date.now()/1000;
function ago(sec){sec=Math.max(0,sec|0);if(sec<60)return sec+"s";if(sec<3600)return (sec/60|0)+"m";return (sec/3600|0)+"h"+((sec%3600)/60|0)+"m";}
function clock(sec){sec=Math.max(0,sec|0);const h=(sec/3600|0),m=((sec%3600)/60|0),s=sec%60;return (h?h+"h":"")+String(m).padStart(h?2:1,"0")+"m"+String(s).padStart(2,"0")+"s";}

const tokBox=$("#tok");
if(tokBox){tokBox.value=localStorage.tok||"";tokBox.addEventListener("change",e=>{localStorage.tok=e.target.value;route();});}
async function api(path,opts){
  const headers={};
  if(localStorage.tok)headers.Authorization="Bearer "+localStorage.tok;
  if(opts&&opts.body)headers["Content-Type"]="application/json";
  for(let a=0;a<2;a++){
    try{const r=await fetch(path,Object.assign({headers},opts));
      if(r.ok)return r.status===204?null:r.json();
      if(r.status<500)throw new Error("http_"+r.status);
    }catch(e){if(a)throw e;}
    await new Promise(res=>setTimeout(res,300));
  }
  throw new Error("request failed");
}

const RUN_ST={completed:"good",running:"run",failed:"bad",cancelled:"bad",queued:"mut",assigned:"mut"};
const pill=(c,l)=>`<span class="st ${c}">${esc(l)}</span>`;
let timers=[];
// viewSeq guards against async renders landing after the user has navigated
// away (clicking a run from the fleet then another from the runs list, etc.).
let viewSeq=0, fleetAll=false, runFilter="all", runLimit=30;
window.setFleetAll=v=>{fleetAll=v;loadOverview();};
window.setRunFilter=s=>{runFilter=s;loadOverview();};
window.moreRuns=()=>{runLimit+=30;loadOverview();};
let me={role:"read",subject:""}, pendingReveal=null;
const ROLES=["read","write","admin","sysadmin"];
const isAdmin=()=>me.role==="admin"||me.role==="sysadmin";
// Access is open to anyone signed in (self-service keys); user management inside
// it stays admin-only. Show the nav once /api/me resolves an identity.
async function loadMe(){let ok=false;try{me=await api("/api/me");ok=!!me.subject;}catch(e){}const a=$("#navAccess");if(a)a.classList.toggle("hidden",!ok);}
function roleOpts(sel){return roleOptsUpTo("sysadmin",sel);}
// Options capped at the caller's own role — you can never grant above yourself.
function roleOptsUpTo(maxRole,sel){const i=ROLES.indexOf(maxRole);const allowed=ROLES.slice(0,i<0?1:i+1);if(!allowed.includes(sel))sel=allowed[allowed.length-1];return allowed.map(r=>`<option ${r===sel?"selected":""}>${r}</option>`).join("");}
function clearTimers(){timers.forEach(clearInterval);timers=[];}
function setClock(ok){const el=$("#clock");if(el)el.textContent=ok?("live · "+new Date().toLocaleTimeString()):"reconnecting…";}

/* ---------------- overview ---------------- */
async function loadOverview(){
  const seq=viewSeq;
  let fleet,runs,spend;
  try{[fleet,runs,spend]=await Promise.all([api("/api/fleet"),api("/api/runs?limit="+runLimit),api("/api/spend")]);}
  catch(e){setClock(false);return;}
  if(seq!==viewSeq)return;
  setClock(true);
  const conn=fleet.filter(w=>w.state==="connected").length;
  const running=runs.filter(r=>r.state==="running").length;
  const queued=runs.filter(r=>r.state==="queued").length;
  const leases=fleet.filter(w=>w.lease&&w.lease.state!=="destroyed").length;
  const tiles=[["v",conn,"Connected <b>workers</b>"],["v",running,"Running <b>runs</b>"],["v",queued,"Queued <b>runs</b>"],
    ["v",leases,"Active <b>leases</b>"],["s","$"+fmt(spend.total_spent||0,4),"Spent <b>· ledger</b>"],
    ["s","$"+fmt(spend.total_committed||0,4),"Committed <b>· leases</b>"]];
  const tilesH=tiles.map(([c,v,l])=>`<div class="tile"><div class="v ${c==='s'?'s':''}">${esc(v)}</div><div class="l">${l}</div></div>`).join("");
  // Fleet: default to active (connected or leased) so stale/disconnected
  // workers don't pile up; "all" reveals them.
  const shownFleet=fleetAll?fleet:fleet.filter(w=>w.state==="connected"||(w.lease&&w.lease.state!=="destroyed"));
  const fleetCtl=`<div class="filters"><button aria-pressed="${!fleetAll}" onclick="setFleetAll(false)">active</button><button aria-pressed="${fleetAll}" onclick="setFleetAll(true)">all</button></div>`;
  const fleetH=shownFleet.length?`<div class="tblwrap"><table><thead><tr><th>worker</th><th>gpu</th><th>state</th><th>hb</th><th>run</th><th>lease</th><th></th></tr></thead><tbody>${shownFleet.map(fleetRow).join("")}</tbody></table></div>`:`<div class="empty">${fleetAll?"no workers":"no active workers"}</div>`;
  // Runs: state filter + "load more" paging (the fetch already grew runLimit).
  const RSTATES=["all","running","queued","completed","failed"];
  const runCtl=`<div class="filters">${RSTATES.map(s=>`<button aria-pressed="${runFilter===s}" onclick="setRunFilter('${s}')">${s}</button>`).join("")}</div>`;
  const shownRuns=runFilter==="all"?runs:runs.filter(r=>r.state===runFilter);
  const more=runs.length>=runLimit?`<button class="more" onclick="moreRuns()">load more…</button>`:"";
  const runsH=shownRuns.length
    ?`<div class="tblwrap"><table><thead><tr><th>id</th><th>name</th><th>kind</th><th>state</th><th>worker</th><th>updated</th></tr></thead><tbody>${shownRuns.map(runRow).join("")}</tbody></table></div>${more}`
    :`<div class="empty">${runFilter==="all"?"no runs yet — submit one via MCP or ./scripts/demo.sh":"no "+runFilter+" runs"}</div>${more}`;
  $("#app").innerHTML=`
    <section><p class="eyebrow">Health</p><div class="tiles">${tilesH}</div></section>
    <section><div class="card"><div class="hd"><h3>Fleet</h3><span class="sp"></span>${fleetCtl}<span class="tag">${shownFleet.length}/${fleet.length}</span></div>${fleetH}</div></section>
    <section><div class="card"><div class="hd"><h3>Runs</h3><span class="sp"></span>${runCtl}</div>${runsH}</div></section>
    <div class="foot">chuk-mcp-training · Neon · R2 hot / Drive archive</div>`;
}
function fleetRow(w){
  const cls=w.state==="connected"?"good":"bad";
  const gpu=(w.hardware&&w.hardware.gpu)||"cpu";
  const vram=w.hardware&&w.hardware.vram_mb?" · "+Math.round(w.hardware.vram_mb/1024)+"GB":"";
  let lease=`<span class="empty" style="padding:0">—</span>`,action="";
  if(w.lease){const L=w.lease,rem=Math.max(0,L.granted_min+(L.extensions||[]).reduce((a,e)=>a+e.minutes,0)-(nows()-L.started_at)/60);
    const lc=L.state==="draining"?"warn":L.state==="destroyed"?"mut":"good";
    lease=`${esc(L.provider)} · $${fmt(L.price_hr)}/h · ${pill(lc,L.state)} · <span class="num">${fmt(rem,1)}m</span>`;
    if(L.state!=="destroyed")action=`<button class="btn" onclick="event.stopPropagation();teardown('${esc(w.id)}')">teardown</button>`;}
  const run=w.current_run?`<a href="#/run/${encodeURIComponent(w.current_run)}" class="mono">${esc(String(w.current_run).slice(0,20))}…</a>`:`<span class="empty" style="padding:0">idle</span>`;
  return `<tr><td class="mono">${esc(w.id)}</td><td>${esc(gpu)}${vram}</td><td>${pill(cls,w.state)}</td><td class="num">${ago(w.heartbeat_age_s)}</td><td>${run}</td><td>${lease}</td><td>${action}</td></tr>`;
}
function runRow(r){
  const cls=RUN_ST[r.state]||"mut";
  return `<tr class="click" onclick="location.hash='#/run/'+encodeURIComponent('${esc(r.id)}')"><td class="mono">${esc(r.id)}</td><td class="name">${esc(r.name)}</td><td>${esc(r.kind)}</td><td>${pill(cls,r.state)}</td><td class="mono">${esc(r.worker_id||"—")}</td><td class="num">${ago(nows()-r.updated_at)} ago</td></tr>`;
}

/* ---------------- run detail ---------------- */
const METRICS=["loss","grad_norm","lr","tokens_per_s","tflops"];
let cur={id:null,metric:"loss",series:{},pinBottom:true};
async function loadRun(id){
  const seq=viewSeq;
  let run;
  try{run=await api("/api/runs/"+encodeURIComponent(id));}catch(e){if(seq!==viewSeq)return;$("#app").innerHTML=backBtn()+`<p class="err">could not load run: ${esc(e.message)}</p>`;return;}
  if(seq!==viewSeq)return;
  cur={id,metric:"loss",series:{},pinBottom:true};
  renderRunShell(run);
  await refreshRun(id,true);
  const el=$("#logs");if(el)el.addEventListener("scroll",()=>{cur.pinBottom=el.scrollTop+el.clientHeight>=el.scrollHeight-24;});
}
function backBtn(){return `<button class="back" onclick="location.hash='#/'">← overview</button>`;}
function specLinks(run){
  const links=run.links||(run.spec&&run.spec.links)||[];
  return links.map(l=>`<a class="olink ${esc(l.kind||'')}" href="${esc(l.url)}" target="_blank" rel="noopener"><span class="g"></span>${esc(l.label||l.url)} ↗</a>`).join("");
}
function renderRunShell(run){
  const cls=RUN_ST[run.state]||"mut";
  const isTrain=run.kind==="train";
  const links=specLinks(run);
  const head=`${backBtn()}
    <div class="runhead" style="margin-top:.9rem">
      <div style="flex:1;min-width:260px">
        <div class="runid">${esc(run.id)}</div>
        <div class="runsub"><span class="name">${esc(run.name)}</span> ${pill(cls,run.state)} <span>${esc(run.kind)}</span>
          <span id="rs-worker">${run.worker_id?"· "+esc(run.worker_id):""}</span>
          <span>· started ${ago(nows()-run.created_at)} ago</span></div>
      </div>
      <div class="links">${links}</div>
      <div class="runacts">${runActions(run)}</div>
    </div>
    <div class="telem" id="telem"></div>`;
  const configCard=`<div class="card"><div class="hd"><h3>Config</h3></div>${configBody(run)}</div>`;
  const body=isTrain?`
    <div class="grid2">
      <div class="stack">
        <div class="card chartcard"><div class="hd"><h3>Metrics</h3><span class="sp"></span><div class="metricsel" id="msel"></div></div>
          <div class="body"><svg class="chart" id="chart" viewBox="0 0 720 210" preserveAspectRatio="none" role="img" aria-label="metric curve"></svg></div></div>
        <div class="card"><div class="hd"><h3>Logs</h3><span class="sp"></span><span class="st ${run.state==='running'?'run':'mut'}" id="logstat">${run.state==='running'?'streaming':run.state}</span></div>
          <div class="logs" id="logs"><div class="empty">loading…</div></div></div>
      </div>
      <div class="stack">
        <div class="card"><div class="hd"><h3>System</h3><span class="sp"></span><span class="tag" id="sysage"></span></div><div class="sys" id="sys"><div class="empty">—</div></div></div>
        ${configCard}
        <div class="card"><div class="hd"><h3>Checkpoints</h3><span class="sp"></span><span class="tag" id="ckcount"></span></div><div id="cks"><div class="empty">—</div></div></div>
        <div class="card"><div class="hd"><h3>Events</h3></div><div class="tl" id="events"></div></div>
      </div>
    </div>`:`
    <div class="grid2"><div class="stack"><div class="card"><div class="hd"><h3>Logs</h3><span class="sp"></span><span class="st mut" id="logstat">${run.state}</span></div><div class="logs" id="logs"><div class="empty">loading…</div></div></div></div>
      <div class="stack">${configCard}<div class="card"><div class="hd"><h3>Events</h3></div><div class="tl" id="events"></div></div></div></div>`;
  $("#app").innerHTML=head+body;
  window.scrollTo(0,0);
}
function configBody(run){
  const s=run.spec||{};
  if(s.kind==="train"){
    const code=s.code?`${esc(s.code.name)}@${esc(String(s.code.sha||"").slice(0,10))}`:"—";
    const ov=(s.overrides&&Object.keys(s.overrides).length)?JSON.stringify(s.overrides):"—";
    return `<dl class="deflist"><dt>code unit</dt><dd>${code}</dd><dt>entrypoint</dt><dd>${esc(s.entrypoint||"—")}</dd>
      <dt>config</dt><dd>${esc(s.config||"—")}</dd><dt>overrides</dt><dd>${esc(ov)}</dd>
      <dt>seed</dt><dd>${esc(s.seed??"—")}</dd><dt>arch</dt><dd>${esc(s.arch||"—")}</dd></dl>`;
  }
  return `<dl class="deflist"><dt>command</dt><dd>${esc(s.command||"—")}</dd><dt>timeout</dt><dd>${esc(s.timeout_s??"—")}s</dd></dl>`;
}
async function refreshRun(id,first){
  const seq=viewSeq;
  let run,metrics,logs,cks,events;
  try{[run,metrics,logs,cks,events]=await Promise.all([
    api("/api/runs/"+encodeURIComponent(id)),
    api("/api/runs/"+encodeURIComponent(id)+"/metrics?keys="+METRICS.join(",")+"&downsample=400").catch(()=>({series:{}})),
    api("/api/runs/"+encodeURIComponent(id)+"/logs?lines=400").catch(()=>({lines:[]})),
    api("/api/runs/"+encodeURIComponent(id)+"/checkpoints").catch(()=>[]),
    api("/api/runs/"+encodeURIComponent(id)+"/events").catch(()=>[]),
  ]);}catch(e){setClock(false);return;}
  if(seq!==viewSeq)return;
  setClock(true);
  cur.series=metrics.series||{};
  renderTelem(run,cks);
  renderMetricSel();
  drawChart();
  renderLogs(logs.lines||logs);
  renderCks(cks,id);
  renderEvents(events);
  const ls=$("#logstat");if(ls){ls.className="st "+(run.state==="running"?"run":"mut");ls.textContent=run.state==="running"?"streaming":run.state;}
  const rw=$("#rs-worker");if(rw)rw.textContent=run.worker_id?"· "+run.worker_id:"";
  // Live host telemetry (GPU/CPU/mem) for the worker running this run.
  if(run.worker_id){
    const t=await api("/api/workers/"+encodeURIComponent(run.worker_id)+"/telemetry").catch(()=>null);
    if(seq===viewSeq)renderSys(t);
  }else renderSys(null);
}
function gbytes(b){return (b/1073741824).toFixed(1);}
function sysBar(frac){const p=Math.round(Math.max(0,Math.min(1,frac||0))*100);return `<div class="bar"><div class="fill" style="width:${p}%"></div></div><span class="pv">${p}%</span>`;}
function renderSys(t){
  const el=$("#sys");if(!el)return;
  const age=$("#sysage");
  const v=t&&t.values;
  if(!v){el.innerHTML=`<div class="empty">no telemetry yet</div>`;if(age)age.textContent="";return;}
  const rows=[],row=(k,inner)=>rows.push(`<div class="sysrow"><div class="sk">${k}</div><div class="sv">${inner}</div></div>`);
  if(v["sys/gpu_util"]!=null)row("GPU",sysBar(v["sys/gpu_util"]));
  if(v["sys/gpu_mem_total_bytes"])row("VRAM",sysBar(v["sys/gpu_mem_used_bytes"]/v["sys/gpu_mem_total_bytes"])+` <small>${gbytes(v["sys/gpu_mem_used_bytes"])}/${gbytes(v["sys/gpu_mem_total_bytes"])} GB</small>`);
  if(v["sys/gpu_temp_c"]!=null)row("Temp",`<b>${Math.round(v["sys/gpu_temp_c"])}</b>°C`);
  if(v["sys/gpu_power_w"]!=null)row("Power",`<b>${Math.round(v["sys/gpu_power_w"])}</b> W`);
  if(v["sys/cpu_util"]!=null)row("CPU",sysBar(v["sys/cpu_util"]));
  if(v["sys/mem_util"]!=null)row("RAM",sysBar(v["sys/mem_util"])+(v["sys/mem_total_bytes"]?` <small>${gbytes(v["sys/mem_used_bytes"])}/${gbytes(v["sys/mem_total_bytes"])} GB</small>`:""));
  el.innerHTML=rows.length?rows.join(""):`<div class="empty">no telemetry yet</div>`;
  if(age)age.textContent=t.sampled_at?ago(nows()-t.sampled_at)+" ago":"";
}
function lastVal(k){const a=cur.series[k];return a&&a.length?a[a.length-1].value:null;}
function maxStep(){let m=0;for(const k in cur.series){const a=cur.series[k];if(a&&a.length)m=Math.max(m,a[a.length-1].step);}return m;}
function renderTelem(run,cks){
  const el=$("#telem");if(!el)return;
  const step=maxStep();
  const lastCk=cks.length?cks[cks.length-1].step:null;
  const elapsed=(run.state==="running"?nows():run.updated_at)-run.created_at;
  const t=[["step",commas(step)],["loss",fmt(lastVal("loss"),3)]];
  const tps=lastVal("tokens_per_s"),tf=lastVal("tflops"),gn=lastVal("grad_norm");
  if(tf!=null)t.push(["throughput",fmt(tf,1)+" <small>TFLOP/s</small>"]);
  if(tps!=null)t.push(["tokens/s",commas(Math.round(tps))]);
  if(gn!=null)t.push(["grad_norm",fmt(gn,3)]);
  t.push(["elapsed",clock(elapsed)]);
  t.push(["last ckpt",lastCk!=null?"step "+commas(lastCk):"—"]);
  el.innerHTML=t.map(([k,v])=>`<div class="t"><div class="k">${esc(k)}</div><div class="val">${v}</div></div>`).join("");
}
function renderMetricSel(){
  const el=$("#msel");if(!el)return;
  const have=METRICS.filter(k=>cur.series[k]&&cur.series[k].length>1);
  if(!have.includes(cur.metric))cur.metric=have[0]||"loss";
  el.innerHTML=have.map(k=>`<button aria-pressed="${k===cur.metric}" onclick="pickMetric('${k}')">${esc(k)}</button>`).join("");
}
window.pickMetric=m=>{cur.metric=m;renderMetricSel();drawChart();};
function drawChart(){
  const svg=$("#chart");if(!svg)return;
  const pts=cur.series[cur.metric]||[];
  if(pts.length<2){svg.innerHTML=`<text x="360" y="105" text-anchor="middle" class="axis">no ${esc(cur.metric)} data yet</text>`;return;}
  const W=720,H=210,pl=48,pr=14,pt=12,pb=22;
  const xs=pts.map(p=>p.step),ys=pts.map(p=>p.value);
  const x0=Math.min(...xs),x1=Math.max(...xs);let y0=Math.min(...ys),y1=Math.max(...ys);
  const pad=(y1-y0)*0.12||Math.abs(y1)*0.1||1;const lo=y0-pad,hi=y1+pad;
  const sx=s=>pl+(x1===x0?0:(s-x0)/(x1-x0))*(W-pl-pr);
  const sy=v=>pt+(1-(v-lo)/(hi-lo))*(H-pt-pb);
  const line=pts.map(p=>`${sx(p.step).toFixed(1)},${sy(p.value).toFixed(1)}`).join(" ");
  const area=`${pl},${H-pb} ${line} ${W-pr},${H-pb}`;
  const exp=cur.metric==="lr";
  let grid="",lab="";
  for(let g=0;g<=3;g++){const v=lo+(hi-lo)*g/3,y=sy(v);
    grid+=`<line x1="${pl}" y1="${y.toFixed(1)}" x2="${W-pr}" y2="${y.toFixed(1)}" stroke="var(--grid)"/>`;
    lab+=`<text x="${pl-6}" y="${(y+3).toFixed(1)}" text-anchor="end" class="axis">${exp?v.toExponential(1):v.toFixed(2)}</text>`;}
  const last=pts[pts.length-1];
  svg.innerHTML=`<defs><linearGradient id="lg" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="var(--accent)" stop-opacity=".28"/><stop offset="1" stop-color="var(--accent)" stop-opacity="0"/></linearGradient></defs>
    ${grid}<polygon points="${area}" fill="url(#lg)"/>
    <polyline points="${line}" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
    <circle cx="${sx(last.step).toFixed(1)}" cy="${sy(last.value).toFixed(1)}" r="3.5" fill="var(--accent)" stroke="var(--plane)" stroke-width="1.5"/>
    ${lab}<text x="${W-pr}" y="${(sy(last.value)-8).toFixed(1)}" text-anchor="end" class="axis" style="fill:var(--ink);font-size:11px">${exp?last.value.toExponential(1):last.value.toFixed(3)}</text>`;
}
function classifyLog(t){if(/checkpoint|ckpt|upload/i.test(t))return"ck";if(/\bstep\b|loss/i.test(t))return"step";if(/error|nan|fail|warn/i.test(t))return"warnln";return"";}
function renderLogs(lines){
  const box=$("#logs");if(!box)return;
  if(!lines||!lines.length){box.innerHTML=`<div class="empty">no logs yet</div>`;return;}
  box.innerHTML=lines.map((l,i)=>`<div class="ln ${classifyLog(l)}"><span class="t">${String(i+1).padStart(4," ")}  </span>${esc(l)}</div>`).join("");
  if(cur.pinBottom)box.scrollTop=box.scrollHeight;
}
function ckLoc(uri){if(/\/ckpt-final\//.test(uri))return"final";if(/\/ckpt-hot\//.test(uri))return"hot";if(/^drive:/.test(uri))return"drive";return"hot";}
function ckKey(uri){return String(uri).replace(/^[a-z0-9]+:\/\/[^/]+\//i,"");}
function renderCks(cks,rid){
  const box=$("#cks");if(!box)return;
  $("#ckcount")&&($("#ckcount").textContent=cks.length?cks.length+" checkpoints":"");
  if(!cks.length){box.innerHTML=`<div class="empty">no checkpoints yet</div>`;return;}
  const rows=cks.slice().reverse().map((c,i)=>{
    const loc=c.location||ckLoc(c.uri);
    const chip=`<span class="chip ${loc}">${loc==='hot'?'R2 · hot':loc==='final'?'R2 · final':'Drive'}</span>`;
    const pin=c.pinned?` <span class="pin" title="pinned">★ ${esc(c.pin_name||"")}</span>`:"";
    const m=c.meta||{};
    const kv=[["seed",m.seed],["arch",m.arch],["config_hash",m.config_hash],["tokenizer_hash",m.tokenizer_hash],
      ["parent",m.parent_checkpoint],["datasets",(m.datasets||[]).join(", ")],["slices",JSON.stringify(m.slices||[])]]
      .filter(([,v])=>v!==undefined&&v!==null&&v!=="").map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
    return `<tr><td class="num">${commas(c.step)}${pin}</td><td>${chip}</td>
      <td><span class="dl" onclick="dl('${esc(rid)}',${c.step},'model.safetensors')">download ↓</span></td>
      <td><span class="expand" onclick="tgl('ckm-${i}',this)">metadata ▸</span></td></tr>
      <tr class="ckmeta hidden" id="ckm-${i}"><td colspan="4"><dl class="kv">${kv||'<dt>—</dt><dd></dd>'}</dl></td></tr>`;
  }).join("");
  box.innerHTML=`<div class="tblwrap"><table><tbody>${rows}</tbody></table></div>`;
}
window.tgl=(id,el)=>{const r=$("#"+id);const open=!r.classList.toggle("hidden");el.textContent=open?"metadata ▾":"metadata ▸";};
// Resolve + download a checkpoint file via the stable endpoint (redirects to R2
// while hot, streams from Drive once archived). Fetch carries the session/token
// auth, so it works in both prod (cookie) and local dev (token box).
window.dl=async (rid,step,file)=>{
  try{
    const h=localStorage.tok?{Authorization:"Bearer "+localStorage.tok}:{};
    const r=await fetch(`/api/checkpoint/${encodeURIComponent(rid)}/${step}/${encodeURIComponent(file)}`,{headers:h});
    if(!r.ok)throw new Error("http_"+r.status);
    const url=URL.createObjectURL(await r.blob());
    const a=document.createElement("a");a.href=url;a.download=`${rid}_step${step}_${file}`;a.click();
    URL.revokeObjectURL(url);
  }catch(e){alert("download failed: "+e.message);}
};
const EV_CLS={running:"run",completed:"good",failed:"bad",cancelled:"bad",checkpoint:"ck"};
function renderEvents(events){
  const box=$("#events");if(!box)return;
  if(!events||!events.length){box.innerHTML=`<div class="empty" style="padding:.4rem 0">no events</div>`;return;}
  box.innerHTML=events.map(e=>{
    const cls=EV_CLS[e.event]||"mut";
    let lbl=esc(e.event);const d=e.detail||{};
    if(d.worker)lbl+=` · <b>${esc(d.worker)}</b>`;
    if(d.step!=null)lbl+=` · step <b>${esc(d.step)}</b>`;
    if(d.exit_code!=null)lbl+=` · exit ${esc(d.exit_code)}`;
    return `<div class="ev ${cls}"><span class="mk"></span><span class="lbl">${lbl}</span><span class="ts">${ago(nows()-e.ts)} ago</span></div>`;
  }).join("");
}
async function teardown(id){
  if(!confirm(`Tear down ${id}? Drains, then destroys the instance (provider-verified).`))return;
  try{await api("/api/workers/"+encodeURIComponent(id)+"/teardown",{method:"POST",body:JSON.stringify({force:false})});route();}
  catch(e){alert("teardown failed: "+e.message);}
}
// Stop is offered while a run is live; resume once it has reached a terminal
// state. The API enforces the write role either way.
function runActions(run){
  const term=["completed","failed","cancelled"].includes(run.state);
  return term
    ? `<button class="btn go" onclick="resumeRun('${esc(run.id)}')">↻ resume</button>`
    : `<button class="btn" onclick="stopRun('${esc(run.id)}')">■ stop</button>`;
}
async function stopRun(id){
  if(!confirm(`Stop ${id}? Signals its worker to cancel the run (it checkpoints, then stops).`))return;
  try{await api("/api/runs/"+encodeURIComponent(id)+"/stop",{method:"POST"});route();}
  catch(e){alert("stop failed: "+e.message);}
}
async function resumeRun(id){
  if(!confirm(`Resume ${id}? Re-queues it; a train run resumes from its latest checkpoint.`))return;
  try{await api("/api/runs/"+encodeURIComponent(id)+"/resume",{method:"POST"});route();}
  catch(e){alert("resume failed: "+e.message);}
}

/* ---------------- access (users + api keys) ---------------- */
async function loadAccess(){
  const seq=viewSeq, admin=isAdmin();
  // Keys are self-service (everyone); the user roster is admin-only, so only
  // fetch it when we can. A key-fetch failure is the real error to surface.
  let keys, users=[];
  try{keys=await api("/api/apikeys");}
  catch(e){$("#app").innerHTML=`<button class="back" onclick="location.hash='#/'">← overview</button><p class="err">Couldn't load your keys (${esc(e.message)}).</p>`;return;}
  if(admin){try{users=await api("/api/users");}catch(e){}}
  if(seq!==viewSeq)return; setClock(true);
  const kRows=keys.map(k=>`<tr${k.revoked_at?' style="opacity:.45"':''}><td>${esc(k.name)}</td><td class="mono">${esc(k.prefix)}…</td>
    <td><span class="rolechip">${esc(k.role)}</span></td><td class="num">${k.last_used_at?ago(nows()-k.last_used_at)+" ago":"never"}</td>
    <td>${k.revoked_at?'<span style="color:var(--muted);font-size:.76rem">revoked</span>':`<span class="rvk" onclick="rmKey('${esc(k.id)}')">revoke</span>`}</td></tr>`).join("");
  const keyDefault=ROLES.includes("write")&&ROLES.indexOf("write")<=ROLES.indexOf(me.role)?"write":me.role;
  const keyCard=`<div class="card"><div class="hd"><h3>API keys</h3><span class="sp"></span><span class="tag">${admin?"team · MCP server":"yours · MCP server"}</span></div>
        <div id="keyBox"></div>
        <div class="tblwrap"><table><thead><tr><th>name</th><th>prefix</th><th>role</th><th>last used</th><th></th></tr></thead><tbody>${kRows||'<tr><td class="empty" colspan="5">no keys yet</td></tr>'}</tbody></table></div>
        <div class="form"><input id="kName" type="text" placeholder="key name (e.g. laptop)"><select id="kRole">${roleOptsUpTo(me.role,keyDefault)}</select><button onclick="mkKey()">generate key</button></div></div>`;
  const expKeyCard=`<div class="card"><div class="hd"><h3>chuk-experiments-server key</h3><span class="sp"></span><span class="tag">${me.experiments_key_set?"linked":"not linked"}</span></div>
        <p class="muted" style="margin:.2rem 0 .6rem">Link your own chuk-experiments-server API key (minted on its own Team screen) so runs you submit report under your identity instead of the shared default.</p>
        <div class="form"><input id="expKey" type="password" placeholder="chuk-experiments-server API key"><button onclick="setExpKey()">save</button>${me.experiments_key_set?'<button onclick="clearExpKey()">clear</button>':""}</div></div>`;
  let usersCard="";
  if(admin){
    const uRows=users.map(u=>`<tr><td class="mono">${esc(u.email)}</td><td><span class="rolechip">${esc(u.role)}</span></td>
      <td>${u.email===me.subject?'<span class="muted" style="color:var(--muted);font-size:.76rem">you</span>':`<span class="rvk" onclick="rmUser('${esc(u.email)}')">remove</span>`}</td></tr>`).join("");
    usersCard=`<div class="card"><div class="hd"><h3>Users</h3><span class="sp"></span><span class="tag">${users.length}</span></div>
        <div class="tblwrap"><table><thead><tr><th>email</th><th>role</th><th></th></tr></thead><tbody>${uRows||'<tr><td class="empty" colspan="3">no users</td></tr>'}</tbody></table></div>
        <div class="form"><input id="uEmail" type="email" placeholder="email@domain"><select id="uRole">${roleOptsUpTo(me.role,"read")}</select><button onclick="addUser()">add / update</button></div></div>`;
  }
  $("#app").innerHTML=`
    <div class="runhead" style="margin-top:.3rem"><div style="flex:1"><div class="runid">Access</div>
      <div class="runsub">${admin?"team members + MCP API keys":"your MCP API keys"} · you are <span class="rolechip">${esc(me.role)}</span></div></div></div>
    <div class="${admin?"grid2":""}">${usersCard}${keyCard}</div>
    <div class="${admin?"grid2":""}">${expKeyCard}</div>
    <div class="foot">roles: read = view · write = submit/manage runs · admin = archive + manage access · sysadmin = all · keys can't exceed your own role</div>`;
  window.scrollTo(0,0);
  if(pendingReveal){$("#keyBox").innerHTML=`<div class="keyreveal"><div class="k">${esc(pendingReveal)}</div><div class="warn">⚠ Copy it now — the key is shown once and only its hash is stored.</div></div>`;pendingReveal=null;}
}
window.addUser=async()=>{const email=($("#uEmail").value||"").trim(),role=$("#uRole").value;if(!email)return;
  try{await api("/api/users",{method:"POST",body:JSON.stringify({email,role})});loadAccess();}catch(e){alert("failed: "+e.message);}};
window.rmUser=async email=>{if(!confirm("Remove "+email+"?"))return;try{await api("/api/users/"+encodeURIComponent(email),{method:"DELETE"});loadAccess();}catch(e){alert("failed: "+e.message);}};
window.mkKey=async()=>{const name=($("#kName").value||"").trim(),role=$("#kRole").value;if(!name)return;
  try{const r=await api("/api/apikeys",{method:"POST",body:JSON.stringify({name,role})});pendingReveal=r.key;loadAccess();}catch(e){alert("failed: "+e.message);}};
window.rmKey=async id=>{if(!confirm("Revoke this key? MCP clients using it stop working immediately."))return;
  try{await api("/api/apikeys/"+encodeURIComponent(id),{method:"DELETE"});loadAccess();}catch(e){alert("failed: "+e.message);}};
window.setExpKey=async()=>{const apiKey=($("#expKey").value||"").trim();if(!apiKey)return;
  try{await api("/api/me/experiments-key",{method:"PUT",body:JSON.stringify({api_key:apiKey})});loadAccess();}catch(e){alert("failed: "+e.message);}};
window.clearExpKey=async()=>{if(!confirm("Clear your linked chuk-experiments-server key?"))return;
  try{await api("/api/me/experiments-key",{method:"DELETE"});loadAccess();}catch(e){alert("failed: "+e.message);}};

/* ---------------- router ---------------- */
function route(){
  clearTimers(); viewSeq++;
  const m=location.hash.match(/#\/run\/(.+)/);
  if(m){const id=decodeURIComponent(m[1]);loadRun(id);timers.push(setInterval(()=>refreshRun(id,false),RUN_MS));}
  else if(location.hash==="#/access"){loadAccess();}
  else{loadOverview();timers.push(setInterval(loadOverview,OVERVIEW_MS));}
}
window.addEventListener("hashchange",route);
setInterval(()=>{const el=$("#clock");if(el&&el.textContent.startsWith("live"))el.textContent="live · "+new Date().toLocaleTimeString();},1000);
loadMe();
route();
</script></body></html>"####;
