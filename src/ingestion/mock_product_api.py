# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Mock Product Catalog REST API (stands in for the catalog microservice).

GET /api/v1/products?page=N&page_size=M   (header X-API-Key required)
GET /health

Behaviour that a real ingestion client must handle:
  * API-key auth (401 without it)
  * pagination with a `next_page` cursor
  * a transient 503 the first time page 3 is requested (tests retry/backoff)
Standard library only, so it runs anywhere.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from src.common import ROLL_NO, SOURCES

API_KEY = f"key-{ROLL_NO}"
HOST, PORT = "127.0.0.1", 8529


class _Handler(BaseHTTPRequestHandler):
    products: list = []
    failed_once: set = set()

    def log_message(self, *args):  # keep console quiet
        pass

    def _send(self, code: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Served-By", f"mock-catalog-{ROLL_NO}")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/health":
            return self._send(200, {"status": "ok"})
        if url.path != "/api/v1/products":
            return self._send(404, {"error": "not found"})
        if self.headers.get("X-API-Key") != API_KEY:
            return self._send(401, {"error": "invalid api key"})
        q = parse_qs(url.query)
        page = int(q.get("page", ["1"])[0])
        size = min(int(q.get("page_size", ["100"])[0]), 200)
        if page == 3 and page not in self.failed_once:
            self.failed_once.add(page)
            return self._send(503, {"error": "service temporarily unavailable"})
        start = (page - 1) * size
        chunk = self.products[start:start + size]
        total = len(self.products)
        nxt = page + 1 if start + size < total else None
        return self._send(200, {"page": page, "page_size": size, "total": total,
                                "next_page": nxt, "data": chunk})


def start_server(port: int = PORT) -> ThreadingHTTPServer:
    with open(SOURCES / "catalog" / "products.json", encoding="utf-8") as f:
        _Handler.products = json.load(f)["products"]
    _Handler.failed_once = set()
    srv = ThreadingHTTPServer((HOST, port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    s = start_server()
    print(f"Mock catalog API on http://{HOST}:{PORT}  (X-API-Key: {API_KEY}). Ctrl+C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        s.shutdown()
