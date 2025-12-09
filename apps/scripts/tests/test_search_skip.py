from pathlib import Path

import apps.scripts.blog_server as blog_server


def test_search_indexes_filters_hits(tmp_path):
    hit = tmp_path / "hit.txt"
    miss = tmp_path / "miss.txt"
    hit.write_text("demo insights are here", encoding="utf-8")
    miss.write_text("nothing to see", encoding="utf-8")

    files = [hit, miss]
    results = blog_server.search_indexes("demo", files)

    assert results == [hit]


def test_main_skips_when_no_hits(monkeypatch, tmp_path):
    logs = []

    monkeypatch.setattr(blog_server, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(blog_server, "KEYWORDS", ["demo"])
    monkeypatch.setattr(blog_server, "init_logger", lambda: None)
    monkeypatch.setattr(blog_server, "log", lambda msg: logs.append(msg))
    monkeypatch.setattr(blog_server, "select_keywords", lambda keywords, seed=None: keywords)
    monkeypatch.setattr(blog_server, "find_index_files", lambda root: [tmp_path / "index.txt"])
    monkeypatch.setattr(blog_server, "search_indexes", lambda kw, files: [])

    blog_server.main()

    assert any("Skipping article generation" in m for m in logs)
