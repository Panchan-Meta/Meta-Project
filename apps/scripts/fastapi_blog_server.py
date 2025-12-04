"""FastAPI server that accepts prompts and optional attachments for blog generation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from blog_builder import DEFAULT_OUTPUT_DIR, STATUS_REPORTER, generate_three_stage_blog

app = FastAPI()

UPLOAD_DIR = Path("/var/www/Meta-Project/data/client")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 転送する添付ファイルの長さを控えめにして、LLMへの入力が肥大化しすぎないようにする
MAX_ATTACHMENT_TEXT_CHARS = 15000
READ_CHUNK_SIZE = 1024 * 1024  # 1 MB


def _read_attachment_text(path: Path) -> str:
    """Read the saved attachment as text, falling back to a permissive decode."""

    try:
        return path.read_text(encoding="utf-8")[:MAX_ATTACHMENT_TEXT_CHARS]
    except UnicodeDecodeError:
        return path.read_bytes().decode("utf-8", errors="ignore")[:MAX_ATTACHMENT_TEXT_CHARS]


async def _save_attachment(attachment: UploadFile) -> tuple[Path, str]:
    """Persist the uploaded file and return its path and readable text content."""

    destination = UPLOAD_DIR / attachment.filename
    try:
        with destination.open("wb") as f:
            while True:
                chunk = await attachment.read(READ_CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
    finally:
        await attachment.close()

    attachment_text = _read_attachment_text(destination)
    return destination, attachment_text


@app.post("/api/generate_blog")
async def generate_blog(
    prompt: str = Form(...),
    attachment: UploadFile | None = File(None),
) -> dict[str, Any]:
    # 添付ファイルがあれば保存し、内容をプロンプトに含める
    prompt_text = prompt.strip()
    saved_file: Path | None = None
    if attachment is not None:
        saved_file, attachment_text = await _save_attachment(attachment)
        attachment_label = f"\n\n[Attachment: {saved_file.name}]\n"
        prompt_text += attachment_label + attachment_text

    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt_missing")

    results = await asyncio.to_thread(generate_three_stage_blog, prompt_text, DEFAULT_OUTPUT_DIR)

    html_map = results.get("html", {}) if isinstance(results, dict) else {}
    files_map = results.get("files", {}) if isinstance(results, dict) else {}

    ja_html = html_map.get("ja", "") if isinstance(html_map, dict) else ""
    ja_file = files_map.get("ja", "") if isinstance(files_map, dict) else ""
    flag = results.get("flag", "FLAG:FILES_SENT") if isinstance(results, dict) else "FLAG:FILES_SENT"

    return {
        "ok": True,
        "html": ja_html,
        "filename": Path(ja_file).name if ja_file else "",
        "flag": flag,
        "files": {code: Path(path).name for code, path in files_map.items()},
        "category": results.get("category") if isinstance(results, dict) else None,
        "status_updates": STATUS_REPORTER.pop_messages(include_current=True),
        "uploaded_file": saved_file.name if saved_file else None,
    }
