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
- 「図解」「ダイアグラム」「diagram」「可視化」といったキーワードを含む図解生成依頼の場合は、自動的にモデルが `codegemma:2b` に切り替わり、JavaScript/CSS を使ったリッチな HTML として生成するよう追加指示を付けて送信します（プロバイダーは `ollama` を想定）。導入・中盤・終盤で 3〜5 個の図解を配置し、記事全体像マップ・概念マップ・フロー／シーケンス図・アーキテクチャ図・比較表・グラフ・リスクマトリクス・要点サマリーなどの候補から選ぶようガイドしています。

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

## generate_blog_from_draft.py

ドラフト (`indexes/mybrain/blog_draft.md`) とインデックス (`indexes/index.json`) を読み込み、
最新記事のメタデータを添えた静的 HTML ブログを出力します。LLM を呼ばずに「ドラフトどおりの体裁」を
一括生成したいときに使います。

### 使い方

```bash
python apps/scripts/generate_blog_from_draft.py \
  --draft indexes/mybrain/blog_draft.md \
  --index indexes/index.json \
  --output /mnt/hgfs/output/blog_auto.html
```

- `--keyword` を指定すると、そのキーワードを含む最新のインデックス記事を優先して参照します。
- キーワード未指定時は、Punk Rock NFT など事前定義の 19 件からランダムで 1 つ選ばれます。
- 出力 HTML にはドラフトのタイトル・ディスクリプション・タグ・各セクション本文・HTML 図解がそのまま反映され、
  選ばれた最新記事（タイトル/URL/要約/公開日時）がメタ情報として表示されます。

## blog_builder.py

ブログ生成フローをまとめた小さなオーケストレータです。現在は `generate_blog_from_draft.py` を内部呼び出しし、
ドラフトから静的 HTML を作ります。サブコマンドを増やす形で他の生成手段を追加できます。

```bash
python apps/scripts/blog_builder.py from_draft \
  --draft indexes/mybrain/blog_draft.md \
  --index indexes/index.json \
  --output /mnt/hgfs/output/blog_auto.html \
  --keyword "Punk Rock NFT"
```

呼び出し結果（生成先パスとキーワード）は標準出力に表示されます。必要に応じてスクリプトから直接
`build_blog_from_draft` 関数を import して使うこともできます。

## ブログ生成フロー

ブログの概論（約 500 文字）とセクション本文（各 1,500 文字）を生成する際のプロンプト設計とデータ参照手順は、クライアントには公開せず `client_instruction_responder.py` 内の `BLOG_WORKFLOW_GUIDE` に埋め込んでいます。`indexes/` 配下のテキストファイル（mybrain 配下を除く）と `indexes/mybrain/` 配下のテキストファイルを組み合わせて LLM に渡すワークフローを確認したいときは、コード内の定数を参照してください。
