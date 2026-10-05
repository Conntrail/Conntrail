"""
Local preview server for the Conntrail website demo.

Serves the static site from this directory and stubs POST /api/contact so the
contact form succeeds locally (it just prints the submission). Use this for a
zero-dependency preview; use `npx wrangler pages dev .` to exercise the real
Cloudflare Pages Function.

    python web/dev_server.py            # http://localhost:8090
    python web/dev_server.py --port 9000
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path.split("?")[0] != "/api/contact":
            self.send_error(404, "not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            data = {"_raw": raw.decode("utf-8", "replace")}
        print("\n[dev_server] contact form submission:")
        print(json.dumps(data, indent=2))
        body = json.dumps({"ok": True}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        # keep the console quiet except for form submissions
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview the Conntrail website demo locally")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    handler = partial(Handler, directory=str(WEB_DIR))
    with ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        print(f"Conntrail demo: http://{args.host}:{args.port}/  (Ctrl+C to stop)")
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
