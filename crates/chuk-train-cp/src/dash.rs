//! Dashboard stub (M0): one page, fleet + runs tables, auto-refresh.
//! Reads the same /api endpoints as the MCP surface; token kept in
//! localStorage and sent as a bearer header.

use axum::response::Html;

const REFRESH_MS: u32 = 3000;

pub async fn page() -> Html<String> {
    Html(PAGE_TEMPLATE.replace("{REFRESH_MS}", &REFRESH_MS.to_string()))
}

const PAGE_TEMPLATE: &str = r#"<!doctype html><html><head><meta charset="utf-8">
<title>chuk-train</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{font-family:ui-monospace,Menlo,monospace;background:#0d1117;color:#c9d1d9;margin:2rem}
 h1{font-size:1.1rem;color:#58a6ff} h2{font-size:.95rem;color:#8b949e;margin-top:1.6rem}
 table{border-collapse:collapse;width:100%;font-size:.85rem}
 td,th{border-bottom:1px solid #21262d;padding:.35rem .6rem;text-align:left}
 .connected,.completed{color:#3fb950}.disconnected,.failed{color:#f85149}
 .running{color:#d29922}.queued,.assigned{color:#8b949e}
 input{background:#161b22;color:#c9d1d9;border:1px solid #30363d;padding:.3rem;width:22rem}
</style></head><body>
<h1>chuk-mcp-training &middot; M0</h1>
<div>token <input id="tok" placeholder="API token" onchange="localStorage.tok=this.value"></div>
<h2>fleet</h2><table id="fleet"></table>
<h2>runs</h2><table id="runs"></table>
<script>
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
document.getElementById('tok').value = localStorage.tok || '';
async function j(p){const r=await fetch(p,{headers:{Authorization:'Bearer '+(localStorage.tok||'')}});
  if(!r.ok)throw new Error(r.status); return r.json();}
async function tick(){try{
 const f=await j('/api/fleet'), r=await j('/api/runs');
 document.getElementById('fleet').innerHTML='<tr><th>worker</th><th>gpu</th><th>state</th><th>hb age</th><th>run</th></tr>'+
  f.map(w=>`<tr><td>${esc(w.id)}</td><td>${esc(w.hardware.gpu||'cpu')}</td><td class="${esc(w.state)}">${esc(w.state)}</td>`+
    `<td>${esc(w.heartbeat_age_s)}s</td><td>${esc(w.current_run||'')}</td></tr>`).join('');
 document.getElementById('runs').innerHTML='<tr><th>id</th><th>name</th><th>state</th><th>worker</th><th>exit</th></tr>'+
  r.map(x=>`<tr><td>${esc(x.id)}</td><td>${esc(x.name)}</td><td class="${esc(x.state)}">${esc(x.state)}</td>`+
    `<td>${esc(x.worker_id||'')}</td><td>${esc(x.exit_code??'')}</td></tr>`).join('');
}catch(e){/* wrong/missing token: leave tables as-is */}}
setInterval(tick,{REFRESH_MS}); tick();
</script></body></html>"#;
