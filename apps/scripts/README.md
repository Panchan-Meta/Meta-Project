# Scripts

## client_instruction_responder.py

クライアントからの指示文を LLM に送り、応答を `/mnt/hgfs/output` 以下に HTML で保存します。

`blog_server.py` からも内部的に呼び出され、フロントエンドのリクエストを処理します。

### 使い方

```bash
python apps/scripts/client_instruction_responder.py "ここに指示文"
```

標準入力からも受け付けます。

```bash
echo "ここに指示文" | python apps/scripts/client_instruction_responder.py
```

オプション:

- `--provider`: `ollama` / `openai` を選択します (デフォルト: `ollama`).
- `--model`: 利用するモデル名を指定します (デフォルト: `phi3:mini`).
- `--api-base`: Ollama 利用時のベース URL (デフォルト: `http://127.0.0.1:11434`).
- `--filename`: 出力ファイル名を指定します。省略時は `client_response_YYYYMMDDTHHMMSSZ.html` 形式になります。

出力先ディレクトリ（`/mnt/hgfs/output`）が存在しない場合でも自動で作成します。

モデル利用の目安:

- ブログのタイトル、200 文字ディスクリプション、6 つのタグ、7 つのセクション名を決めるとき: `phi3:mini` (Ollama)
- セクション本文や図解作成、全文要約（総論）: `llama3:8b` (Ollama)

OpenAI を使う場合は Python SDK (`openai` パッケージ) と API キーが必要です。依存パッケージがない場合は HTML 内に警告が表示されます。

## blog_server.py

フロントエンドからの `/api/generate_blog` (および互換の `/api/generate_content_plan`) リクエストを受け取り、
`client_instruction_responder.py` を経由して LLM 応答を HTML として生成し保存します。サーバー起動例:

```bash
python apps/scripts/blog_server.py
```

JSON リクエスト例:

```json
{
  "prompt": "ここに指示文",
  "model": "phi3:mini",
  "provider": "ollama",
  "api_base": "http://127.0.0.1:11434",
  "filename": "任意のファイル名.html"
}
```

## ブログ生成フロー

ブログの概論（約 500 文字）とセクション本文（各 1,500 文字）を生成する際のプロンプト設計とデータ参照手順は、クライアントには公開せず `client_instruction_responder.py` 内の `BLOG_WORKFLOW_GUIDE` に埋め込んでいます。`indexes/` 配下のテキストファイル（mybrain 配下を除く）と `indexes/mybrain/` 配下のテキストファイルを組み合わせて LLM に渡すワークフローを確認したいときは、コード内の定数を参照してください。
