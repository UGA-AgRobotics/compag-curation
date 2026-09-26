"""Loopback web form: select the original Only_codes roots once and import them."""

from __future__ import annotations

import hmac
import html
import json
import secrets
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

_FORM = """<!doctype html><html><head><meta charset="utf-8"><title>Import Only_codes workspace</title>
<style>body{{font:14px system-ui;background:#0b1320;color:#eaf2ff;margin:24px}}label{{display:block;margin:10px 0 3px}}
input,textarea{{width:720px;max-width:95vw;padding:6px;background:#111c2e;color:#eaf2ff;border:1px solid #2a3d5c;border-radius:5px}}
button{{margin-top:14px;padding:8px 14px}}pre{{background:#111c2e;padding:10px;white-space:pre-wrap}}</style></head><body>
<h2>Import an Only_codes workspace (profile only-codes-compat-v1)</h2>
<p>Originals are read-only. Every update is written to the new project directory with the legacy relative layout.</p>
<form method="post" action="/import?token={token}">
<label>Legacy root (the old <code>Clean</code> directory)</label><input name="legacy_root" required>
<label>SAM2 repository root (contains checkpoints/sam2.1_hiera_large.pt)</label><input name="sam2_repo_root" required>
<label>TORCH_HOME (contains hub/checkpoints/resnet50-11ad3fa6.pth)</label><input name="torch_home" required>
<label>Current round image folder (optional, e.g. IMG_9510)</label><input name="image">
<label>Path prefix mappings, one per line: OLD_PREFIX=NEW_PREFIX</label><textarea name="maps" rows="3"></textarea>
<label>New project directory (must not exist)</label><input name="output" required>
<button type="submit">Import</button></form>{result}</body></html>"""


class _H(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        return

    def _ok(self, q):
        return hmac.compare_digest((q.get("token") or [""])[0], self.server.token) and \
            (self.headers.get("Host") or "").split(":")[0] in ("127.0.0.1", "localhost")

    def _page(self, result=""):
        body = _FORM.format(token=self.server.token, result=result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        q = parse_qs(urlsplit(self.path).query)
        if not self._ok(q):
            self.send_error(403)
            return
        self._page()

    def do_POST(self):  # noqa: N802
        q = parse_qs(urlsplit(self.path).query)
        if not self._ok(q):
            self.send_error(403)
            return
        n = int(self.headers.get("Content-Length") or 0)
        form = parse_qs(self.rfile.read(min(n, 65536)).decode("utf-8"))
        g = lambda k: (form.get(k) or [""])[0].strip()  # noqa: E731
        try:
            from .project import import_project

            maps = []
            for line in g("maps").splitlines():
                if line.strip():
                    o, nn = line.split("=", 1)
                    maps.append((o.strip(), nn.strip()))
            proj = import_project(legacy_root=Path(g("legacy_root")), output=Path(g("output")),
                                  sam2_repo_root=Path(g("sam2_repo_root")), torch_home=Path(g("torch_home")),
                                  image=g("image") or None, path_mappings=maps)
            rows = proj.manifest["inventory"]
            res = {"status": "PASS", "project": str(proj.root),
                   "missing": [r["input_type"] for r in rows if r["verification_status"] == "MISSING"],
                   "next": "convert the round model: compag-curation only-codes import-model --project ... "
                           "--round-dir rNN_hybrid --trust-legacy-pickle <sha256>"}
        except Exception as exc:
            res = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
        self._page("<pre>" + html.escape(json.dumps(res, indent=2)) + "</pre>")


def run_import_ui(*, port: int = 0, open_browser: bool = True) -> int:
    srv = ThreadingHTTPServer(("127.0.0.1", int(port)), _H)
    srv.token = secrets.token_urlsafe(24)
    url = f"http://127.0.0.1:{srv.server_address[1]}/?token={srv.token}"
    print(json.dumps({"status": "SERVING", "url": url}))
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
