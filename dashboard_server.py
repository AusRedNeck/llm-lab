#!/usr/bin/env python3
"""Quick dashboard server for training monitor."""
import json
import os
import http.server
import socketserver

LOSS_FILE = "runs/20260915_1437_bpe50m_rope_synthetic/loss.jsonl"
PORT = 8769

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/loss":
            try:
                with open(LOSS_FILE) as f:
                    lines = f.readlines()
                last = json.loads(lines[-1])
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(last).encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode())
        elif self.path == "/" or self.path == "/index.html":
            self.path = "/dashboard.html"
            super().do_GET()
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass  # silent

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Dashboard: http://localhost:{PORT}")
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
