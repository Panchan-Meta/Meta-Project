"""Minimal HTTP server to receive client prompts and return generated blog HTML."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from blog_builder import DEFAULT_OUTPUT_DIR, generate_blogs


class BlogRequestHandler(BaseHTTPRequestHandler):
    server_version = "BlogHTMLServer/1.0"

    def _read_body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length) if length else b""
        try:
            return json.loads(data.decode("utf-8")) if data else {}
        except json.JSONDecodeError:
            return {}

    def _send_json(self, payload: dict[str, object], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 (HTTP verb casing)
        if self.path != "/api/generate_blog":
            self._send_json({"ok": False, "error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return

        data = self._read_body()
        prompt = str(data.get("prompt", "")).strip() if isinstance(data, dict) else ""
        if not prompt:
            self._send_json({"ok": False, "error": "prompt_missing"}, status=HTTPStatus.BAD_REQUEST)
            return

        results = generate_blogs(prompt, output_dir=DEFAULT_OUTPUT_DIR)
        html_map = results.get("html", {}) if isinstance(results, dict) else {}
        files_map = results.get("files", {}) if isinstance(results, dict) else {}

        ja_html = html_map.get("ja", "") if isinstance(html_map, dict) else ""
        ja_file = files_map.get("ja", "") if isinstance(files_map, dict) else ""
        flag = results.get("flag", "FLAG:FILES_SENT") if isinstance(results, dict) else "FLAG:FILES_SENT"

        response = {
            "ok": True,
            "html": ja_html,
            "filename": Path(ja_file).name if ja_file else "",
            "flag": flag,
            "files": {code: Path(path).name for code, path in files_map.items()},
            "category": results.get("category") if isinstance(results, dict) else None,
        }
        self._send_json(response)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = HTTPServer((host, port), BlogRequestHandler)
    print(f"Serving blog generator on http://{host}:{port}/api/generate_blog")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
