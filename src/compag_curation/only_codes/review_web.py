"""Loopback-only browser reviewer for legacy (Only_codes) review sessions.

Presentation is new; state semantics come from :class:`review.LegacyReviewSession`
(the ported ``full_image_review_gui.py`` logic).  The page shows the original
candidate geometry (polygon or recorded bbox rectangle), the GUI decision,
probabilities, thresholds, human labels and the pipeline's AL shortlist.
"""

from __future__ import annotations

import csv
import hmac
import json
import secrets
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

from .review import ACTIONS, LegacyReviewSession

_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><title>Only_codes compatible review</title>
<style>
:root{color-scheme:dark;--bg:#0b1320;--panel:#111c2e;--line:#2a3d5c;--text:#eaf2ff;--muted:#9fb2cc}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 system-ui,sans-serif}
header{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px 14px;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:15px;margin:0 12px 0 0}select,button,label{font:inherit}button{background:#1b2a44;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px 9px;cursor:pointer}
button:hover{background:#243759}.main{display:grid;grid-template-columns:1fr 360px;height:calc(100vh - 54px)}
#wrap{position:relative;overflow:hidden;background:#000}canvas{position:absolute;left:0;top:0}
aside{border-left:1px solid var(--line);background:var(--panel);overflow:auto;padding:10px}
.mono{font-family:ui-monospace,monospace;white-space:pre-wrap;font-size:12px}.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:3px 4px;text-align:left;font-size:12px}
tr.sel{background:#29406a}.badge{display:inline-block;padding:1px 6px;border-radius:9px;background:#223}
</style></head><body>
<header><h1>Only_codes review <span class="badge">only-codes-compat-v1</span></h1>
<label>Base <select id="base"></select></label>
<label>Filter <select id="filter"><option value="all">All</option><option value="uncertain">Uncertain</option>
<option value="certain">Certain</option><option value="cj">CJ</option><option value="noncj">Non-CJ</option>
<option value="reviewed">Reviewed</option><option value="unreviewed">Unreviewed</option></select></label>
<label><input type="checkbox" id="hide_rev">Hide reviewed</label><label><input type="checkbox" id="hide_seen" checked>Hide seen</label>
<label><input type="checkbox" id="mismatch">Only mismatches</label><label><input type="checkbox" id="auto_next" checked>Auto-next</label>
<button data-a="accept">1 Accept</button><button data-a="flip">2 Flip</button><button data-a="skip">3 Skip</button>
<button data-a="sus_accept">4 Sus-accept</button><button data-a="sus_flip">5 Sus-flip</button>
<button data-a="auto_accept_rest">Accept remaining (base)</button><button data-a="unsee">Unsee</button><button data-a="undo">0 Undo</button>
<span class="muted">Wheel: zoom · drag: pan</span>
</header>
<div class="main"><div id="wrap"><canvas id="cv"></canvas></div>
<aside><div id="status" class="mono muted"></div><h3>Selected</h3><div id="sel" class="mono">Click a candidate.</div>
<div id="msg" class="mono"></div><h3>AL shortlist (pipeline al_candidates.csv)</h3><table id="al"></table></aside></div>
<script nonce="__NONCE__">
const TOKEN="__TOKEN__";let S=null,items=[],base=null,img=new Image(),sel=null,view={s:1,x:0,y:0},drag=null;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d'),wrap=document.getElementById('wrap');
function api(p,o){o=o||{};o.headers=Object.assign({'X-Review-Token':TOKEN,'Content-Type':'application/json'},o.headers||{});return fetch(p,o).then(r=>r.json())}
function visible(it){const o=opts();if(o.hide_rev&&it.reviewed)return false;if(o.hide_seen&&it.seen)return false;
 if(o.mismatch)return it.mismatch;let dp=it.pred,du=it.uncertain;if(it.reviewed&&(it.human===0||it.human===1)){dp=it.human;du=false}
 const f=document.getElementById('filter').value;if(f==='uncertain'&&!du)return false;if(f==='certain'&&du)return false;
 if(f==='cj'&&dp!==1)return false;if(f==='noncj'&&dp!==0)return false;if(f==='reviewed'&&!it.reviewed)return false;if(f==='unreviewed'&&it.reviewed)return false;return true}
function opts(){return{hide_rev:hide_rev.checked,hide_seen:hide_seen.checked,mismatch:mismatch.checked,auto_next:auto_next.checked}}
function cen(it){const xs=it.xs.slice(0,-1),ys=it.ys.slice(0,-1);return[(Math.min(...xs)+Math.max(...xs))/2,(Math.min(...ys)+Math.max(...ys))/2]}
function draw(){cv.width=wrap.clientWidth;cv.height=wrap.clientHeight;ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='#000';ctx.fillRect(0,0,cv.width,cv.height);
 ctx.setTransform(view.s,0,0,view.s,view.x,view.y);if(img.complete)ctx.drawImage(img,0,0);
 for(const it of items){if(!visible(it))continue;let col=it.uncertain?'#ffd400':(it.pred===1?'#39ff14':'#ff3b3b');let dash=[];
  if(it.reviewed&&(it.human===0||it.human===1)){col=it.human===1?'#39ff14':'#ff3b3b';dash=[6,4]}if(it.mismatch)col='#ff00ff';if(sel===it.key)col='#ffffff';
  ctx.setLineDash(dash.map(d=>d/view.s));ctx.lineWidth=(sel===it.key?4:2)/view.s;ctx.strokeStyle=col;ctx.beginPath();
  for(let i=0;i<it.xs.length;i++){i?ctx.lineTo(it.xs[i],it.ys[i]):ctx.moveTo(it.xs[i],it.ys[i])}ctx.stroke();ctx.setLineDash([]);
  const c=cen(it);ctx.fillStyle=it.reviewed?col:(it.uncertain?'#ffd400':col);ctx.beginPath();ctx.arc(c[0],c[1],(it.reviewed?5:3)/view.s,0,7);ctx.fill()}}
function fmt(v){return v===null||v===undefined?'NA':(typeof v==='number'?v.toFixed(3):String(v))}
function showSel(){const it=items.find(i=>i.key===sel);if(!it){document.getElementById('sel').textContent='Click a candidate.';return}
 document.getElementById('sel').textContent=`key=${it.key}\nGUI pred=${it.pred?'CJ':'Non-CJ'}${it.uncertain?' (uncertain)':''} p=${fmt(it.p_ui)} thr=${fmt(it.thr_eff)}\n`+
 `xgb_p=${fmt(it.xgb_p)} yolo=${fmt(it.yolo_conf)}/${fmt(it.yolo_iou)} p_fused=${fmt(it.p_fused)} pipeline_kept=${it.kept_pipeline}\n`+
 `geometry=${it.geometry}\nreviewed=${it.reviewed} human=${it.human} action=${it.action||''} seen=${it.seen} mismatch=${it.mismatch}`}
function load(){api('/api/state?base='+encodeURIComponent(base||'')).then(d=>{S=d;base=d.base;items=d.items;
 const bs=document.getElementById('base');if(!bs.options.length){for(const b of d.bases){const o=document.createElement('option');o.value=o.textContent=b;bs.appendChild(o)}}bs.value=base;
 document.getElementById('status').textContent=`policy=${d.params.det_policy}/${d.params.det_missing} thr_xgb=${d.params.thr_xgb} thr_yolo=${d.params.thr_yolo}/${d.params.thr_iou} det_thr=${d.params.det_thr} margin=${d.params.al_margin}\nlabels=${d.n_labels} seen=${d.n_seen} rows(base)=${items.length}\nreview_csv=${d.review_csv}`;
 const al=document.getElementById('al');al.innerHTML='<tr><th>image</th><th>id</th><th>score</th><th>reason</th><th>p</th><th>thr</th></tr>';
 for(const r of d.al){const tr=document.createElement('tr');tr.innerHTML=`<td>${r.image.replace(/^.*_(y\d+x\d+).*$/,'$1')}</td><td>${r.id}</td><td>${fmt(+r.al_score)}</td><td>${r.al_reason||''}</td><td>${fmt(+r.xgb_proba)}</td><td>${r.thr_eff||''}</td>`;
  tr.onclick=()=>{sel=r.image+'||'+r.id;showSel();draw()};if(sel===r.image+'||'+r.id)tr.className='sel';al.appendChild(tr)}
 if(!img.src||img.dataset.base!==base){img=new Image();img.dataset.base=base;img.onload=()=>{fit();draw()};img.src='/mosaic?base='+encodeURIComponent(base)+'&token='+TOKEN}else draw();showSel()})}
function fit(){const names=[...new Set(items.map(it=>it.image))];
 if(names.length===1){const m=names[0].match(/_y(\d+)x(\d+)\.(?:jpg|jpeg|png)$/i);
  if(m){const y=+m[1],x=+m[2],s=Math.min(wrap.clientWidth/512,wrap.clientHeight/512);
   view={s:s,x:(wrap.clientWidth-512*s)/2-x*s,y:(wrap.clientHeight-512*s)/2-y*s};return}}
 const s=Math.min(wrap.clientWidth/img.width,wrap.clientHeight/img.height);view={s:s,x:0,y:0}}
function nearestAfter(k){const it=items.find(i=>i.key===k);if(!it)return null;const c0=cen(it);let best=null,bd=1e18;
 for(const o of items){if(o.key===k||!visible(o))continue;const c=cen(o),d=(c[0]-c0[0])**2+(c[1]-c0[1])**2;if(d<bd){bd=d;best=o.key}}return best}
function act(a){api('/api/action',{method:'POST',body:JSON.stringify({action:a,key:sel,base:base})}).then(d=>{document.getElementById('msg').textContent=d.msg||d.error||'';
 const auto=['accept','flip','sus_accept','sus_flip','skip'].includes(a)&&auto_next.checked;const cur=sel;load();if(auto){setTimeout(()=>{const n=nearestAfter(cur);if(n){sel=n;showSel();draw()}},250)}})}
document.querySelectorAll('button[data-a]').forEach(b=>b.onclick=()=>act(b.dataset.a));
['filter','hide_rev','hide_seen','mismatch'].forEach(id=>document.getElementById(id).onchange=draw);
document.getElementById('base').onchange=e=>{base=e.target.value;sel=null;load()};
cv.onwheel=e=>{e.preventDefault();const f=e.deltaY<0?1.15:1/1.15;const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;view.x=mx-(mx-view.x)*f;view.y=my-(my-view.y)*f;view.s*=f;draw()};
cv.onmousedown=e=>{drag={x:e.clientX,y:e.clientY,vx:view.x,vy:view.y,moved:false}};
window.onmousemove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>3)drag.moved=true;view.x=drag.vx+dx;view.y=drag.vy+dy;draw()};
window.onmouseup=e=>{if(drag&&!drag.moved){const r=cv.getBoundingClientRect(),px=(e.clientX-r.left-view.x)/view.s,py=(e.clientY-r.top-view.y)/view.s;let best=null,bd=1e18;
 for(const it of items){if(!visible(it))continue;const c=cen(it),d=(c[0]-px)**2+(c[1]-py)**2;if(d<bd){bd=d;best=it.key}}if(best&&bd<(40/view.s)**2){sel=best;showSel();draw()}}drag=null};
window.onkeydown=e=>{if(e.ctrlKey||e.metaKey||e.altKey)return;const k=e.key.toLowerCase();const m={'1':'accept','a':'accept','2':'flip','r':'flip','3':'skip','s':'skip','4':'sus_accept','w':'sus_accept','5':'sus_flip','u':'sus_flip','0':'undo','z':'undo'};if(m[k]){e.preventDefault();act(m[k])}};
window.onresize=draw;load();
</script></body></html>"""


def _read_al(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler, session: LegacyReviewSession, token: str, on_action):
        super().__init__(addr, handler)
        self.session = session
        self.token = token
        self.nonce = secrets.token_urlsafe(16)
        self.lock = threading.Lock()
        self.on_action = on_action


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, *_a):  # quiet
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if ctype.startswith("text/html"):
            self.send_header("Content-Security-Policy",
                             f"default-src 'none'; img-src 'self'; connect-src 'self'; style-src 'unsafe-inline'; "
                             f"script-src 'nonce-{self.server.nonce}'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode("utf-8"), "application/json")

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def _token_ok(self, q: dict) -> bool:
        tok = self.headers.get("X-Review-Token") or (q.get("token") or [""])[0]
        return hmac.compare_digest(str(tok), self.server.token)

    def do_GET(self):  # noqa: N802
        u = urlsplit(self.path)
        q = parse_qs(u.query)
        if not self._host_ok() or not self._token_ok(q):
            return self._send(403, b"forbidden", "text/plain")
        sess = self.server.session
        if u.path == "/":
            if getattr(sess, "project_ui", False):
                from ..r92_review_ui import HTML as template
            else:
                template = _HTML
            html = template.replace("__TOKEN__", self.server.token).replace("__NONCE__", self.server.nonce)
            return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        if u.path == "/api/state":
            base = (q.get("base") or [""])[0] or sess.bases[0]
            if base not in sess.bases:
                return self._json({"error": "unknown base"}, 400)
            with self.server.lock:
                items = sess.items(base)
                if getattr(sess, "project_ui", False):
                    al = [{"image": k.split("||", 1)[0], "id": k.split("||", 1)[1], "rank": rank}
                          for k, rank in sorted(sess.project_selection.items(), key=lambda pair: pair[1])]
                else:
                    al = [r for r in _read_al(sess.detections_csv.parent / "al_candidates.csv")]
                return self._json({"base": base, "bases": sess.bases, "items": items, "params": sess.params.__dict__,
                                   "n_labels": len(sess.labels_eff), "n_seen": len(sess.seen),
                                   "review_csv": str(sess.review_csv), "al": al,
                                   "pool_count": getattr(sess, "project_pool_count", len(sess.df)),
                                   "shortlist_count": len(getattr(sess, "project_selection", {})),
                                   "project_ui": bool(getattr(sess, "project_ui", False))})
        if u.path == "/api/bulk-plan" and getattr(sess, "project_ui", False):
            with self.server.lock:
                return self._json(sess.bulk_plan())
        if u.path == "/mosaic":
            base = (q.get("base") or [""])[0]
            if base not in sess.bases:
                return self._send(404, b"", "text/plain")
            with self.server.lock:
                path, _, _ = sess.mosaic(base)
            return self._send(200, path.read_bytes(), "image/jpeg")
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        u = urlsplit(self.path)
        if not self._host_ok() or not self._token_ok({}):
            return self._send(403, b"forbidden", "text/plain")
        if u.path != "/api/action":
            return self._send(404, b"", "text/plain")
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 65536:
            return self._json({"error": "bad body"}, 400)
        try:
            body = json.loads(self.rfile.read(n))
            action = str(body.get("action"))
            project_actions = {"delete", "accept_remaining", "undo_bulk"} if getattr(self.server.session, "project_ui", False) else set()
            if action not in ACTIONS and action not in project_actions:
                raise ValueError("unknown action")
            with self.server.lock:
                if action == "accept_remaining":
                    msg = self.server.session.act(action, expected_count=body.get("expected_count"),
                                                  expected_log_sha256=body.get("expected_log_sha256"))
                else:
                    msg = self.server.session.act(action, sel=body.get("key"), base=body.get("base"))
            if self.server.on_action:
                self.server.on_action(action, body.get("key"))
            return self._json({"msg": msg})
        except Exception as exc:
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)


def create_server(session: LegacyReviewSession, *, port: int = 0,
                  on_action: Callable[[str, Any], None] | None = None) -> tuple[_Server, str]:
    token = secrets.token_urlsafe(24)
    srv = _Server(("127.0.0.1", int(port)), _Handler, session, token, on_action)
    url = f"http://127.0.0.1:{srv.server_address[1]}/?token={token}"
    return srv, url


def run_review_web(session: LegacyReviewSession, *, port: int = 0, open_browser: bool = True,
                   on_action: Callable[[str, Any], None] | None = None) -> int:
    srv, url = create_server(session, port=port, on_action=on_action)
    print(json.dumps({"status": "SERVING", "url": url, "review_csv": str(session.review_csv),
                      "hint": "Ctrl+C to stop; every action is appended to review_labels.csv immediately"}))
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0
