#!/usr/bin/env python3
"""HTTP server for Medical Concept Labeling Tool UI."""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

HOST = "127.0.0.1"
PORT = 9090
UI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(UI_DIR)
INPUT_DIR = os.path.join(PROJECT_DIR, "input")
GOLD_DIR = os.path.join(PROJECT_DIR, "pipeline", "output")

os.makedirs(GOLD_DIR, exist_ok=True)

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, content, content_type="application/json; charset=utf-8"):
        body = json.dumps(content, ensure_ascii=False).encode("utf-8") if isinstance(content, (dict, list)) else content
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, path):
        filepath = os.path.join(UI_DIR, path.lstrip("/"))
        if not os.path.isfile(filepath):
            self.send_error(404)
            return
        ext = os.path.splitext(filepath)[1]
        ct = MIME.get(ext, "application/octet-stream")
        with open(filepath, "rb") as f:
            self._send(200, f.read(), ct)

    def do_GET(self):
        p = urlparse(self.path).path

        if p == "/" or p == "/index.html":
            return self._serve_static("/index.html")
        if p.endswith((".html", ".css", ".js")):
            return self._serve_static(p)

        if p == "/api/files":
            files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")],
                           key=lambda x: int(x.replace(".txt", "")))
            return self._send(200, files)

        if p.startswith("/api/input/"):
            fid = p.split("/")[-1]
            path = os.path.join(INPUT_DIR, f"{fid}.txt")
            if not os.path.exists(path):
                return self._send(404, {"error": "not found"})
            with open(path, "r", encoding="utf-8") as f:
                return self._send(200, {"id": fid, "text": f.read()})

        if p.startswith("/api/gold/"):
            fid = p.split("/")[-1]
            path = os.path.join(GOLD_DIR, f"{fid}.json")
            if not os.path.exists(path):
                return self._send(200, [])
            with open(path, "r", encoding="utf-8") as f:
                try:
                    return self._send(200, json.load(f))
                except json.JSONDecodeError:
                    return self._send(200, [])

        if p == "/api/export":
            result = {}
            for f in os.listdir(GOLD_DIR):
                if f.endswith(".json"):
                    fid = f.replace(".json", "")
                    with open(os.path.join(GOLD_DIR, f), "r", encoding="utf-8") as fh:
                        result[fid] = json.load(fh)
            return self._send(200, result)

        self.send_error(404)

    def do_PUT(self):
        p = urlparse(self.path).path
        if p.startswith("/api/gold/"):
            fid = p.split("/")[-1]
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            data = json.loads(body)
            path = os.path.join(GOLD_DIR, f"{fid}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return self._send(200, {"ok": True, "count": len(data) if isinstance(data, list) else 0})
        self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        pass  # quiet


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), Handler)
    print(f"Labeling Tool: http://{HOST}:{PORT}")
    print(f"Input: {INPUT_DIR}")
    print(f"Gold:  {GOLD_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
