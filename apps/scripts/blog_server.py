"""Minimal HTTP server to receive client prompts and return generated blog HTML."""

from __future__ import annotations

import cgi
import io
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from blog_builder import DEFAULT_OUTPUT_DIR, STATUS_REPORTER, generate_three_stage_blog

UPLOAD_DIR = Path("/var/www/Meta-Project/data/client")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_ATTACHMENT_TEXT_CHARS = 15000
READ_CHUNK_SIZE = 1024 * 1024  # 1 MB


class BlogRequestHandler(BaseHTTPRequestHandler):
    server_version = "BlogHTMLServer/1.0"

    def _read_attachment_text(self, path: Path) -> str:
        """Read a stored attachment as text with a lenient decoder."""
        try:
            return path.read_text(encoding="utf-8")[:MAX_ATTACHMENT_TEXT_CHARS]
        except UnicodeDecodeError:
            return path.read_bytes().decode("utf-8", errors="ignore")[:MAX_ATTACHMENT_TEXT_CHARS]

    def _save_attachment(self, field: cgi.FieldStorage) -> tuple[Path, str]:
        """Persist an uploaded file field and return its path and readable text content."""
        filename = field.filename or "upload.bin"
        destination = UPLOAD_DIR / filename

        # FieldStorage keeps the file handle open on .file
        with destination.open("wb") as f:
            while True:
                chunk = field.file.read(READ_CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)

        attachment_text = self._read_attachment_text(destination)
        return destination, attachment_text

    def _build_prompt_from_multipart(self, form: cgi.FieldStorage) -> tuple[str, list[Path]]:
        """Compose the prompt including any attachment content from multipart data."""
        prompt_text = str(form.getfirst("prompt", "")).strip()
        saved_files: list[Path] = []

        attachments = form.getlist("attachment") if "attachment" in form else []
        # FieldStorage may return a single item when only one file is uploaded.
        if isinstance(attachments, cgi.FieldStorage):
            attachments = [attachments]

        for attachment in attachments:
            if not isinstance(attachment, cgi.FieldStorage) or not attachment.filename:
                continue
            saved_path, attachment_text = self._save_attachment(attachment)
            saved_files.append(saved_path)
            attachment_label = f"\n\n[Attachment: {saved_path.name}]\n"
            prompt_text += attachment_label + attachment_text

        return prompt_text, saved_files

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

    # --------------------------------------------------------------------- #
    #  POST /api/generate_blog
    # --------------------------------------------------------------------- #
    def do_POST(self) -> None:  # noqa: N802 (HTTP verb casing)
        if self.path != "/api/generate_blog":
            self._send_json({"ok": False, "error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return

        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", "0"))
        uploaded_files: list[Path] = []

        # 1) 入力の取り出し（プレーン JSON or multipart/form-data）
        if content_type.startswith("multipart/form-data"):
            form = cgi.FieldStorage(
                fp=io.BytesIO(self.rfile.read(length)),
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                },
            )
            prompt_text, uploaded_files = self._build_prompt_from_multipart(form)
        else:
            data = self._read_body()
            prompt_text = str(data.get("prompt", "")).strip() if isinstance(data, dict) else ""

        if not prompt_text:
            self._send_json({"ok": False, "error": "prompt_missing"}, status=HTTPStatus.BAD_REQUEST)
            return

        # 2) ブログ生成本体
        results = generate_three_stage_blog(prompt_text, output_dir=DEFAULT_OUTPUT_DIR)
        if not isinstance(results, dict):
            self._send_json({"ok": False, "error": "generation_failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        html_map = results.get("html", {}) if isinstance(results, dict) else {}
        files_map = results.get("files", {}) if isinstance(results, dict) else {}

        ja_html = html_map.get("ja", "") if isinstance(html_map, dict) else ""
        ja_file = files_map.get("ja", "") if isinstance(files_map, dict) else ""
        flag = results.get("flag", "FLAG:FILES_SENT")

        # ファイルがあればディスク上の内容を優先
        response_html = ja_html
        if ja_file and Path(ja_file).is_file():
            response_html = Path(ja_file).read_text(encoding="utf-8")


        if not response_html:
            self._send_json(
                {"ok": False, "error": "generation_failed"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        filename = Path(ja_file).name if ja_file else "blog.html"

        # クライアントがそのまま json.loads / resp.json() できる形で返す
        payload: dict[str, object] = {
            "ok": True,
            "flag": flag,
            "filename": filename,
            "html": response_html,  # ここに HTML 本文を全部入れる
            "files": {code: Path(path).name for code, path in files_map.items()},
            "category": results.get("category") if isinstance(results, dict) else None,
            "status_updates": STATUS_REPORTER.pop_messages(include_current=True),
            "uploaded_files": [path.name for path in uploaded_files],
        }
        self._send_json(payload, status=HTTPStatus.OK)


    # --------------------------------------------------------------------- #
    #  GET /api/status
    # --------------------------------------------------------------------- #
    def do_GET(self) -> None:  # noqa: N802 (HTTP verb casing)
        if self.path != "/api/status":
            self._send_json({"ok": False, "error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return

        updates = STATUS_REPORTER.pop_messages(include_current=True)
        self._send_json({"ok": True, "updates": updates})


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = HTTPServer((host, port), BlogRequestHandler)
    print(f"Serving blog generator on http://{host}:{port}/api/generate_blog")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
