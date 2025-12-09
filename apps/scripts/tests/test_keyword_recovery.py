import apps.scripts.blog_server as blog_server


def test_search_indexes_by_parts_hits_partial(tmp_path):
    hit = tmp_path / "hit.txt"
    hit.write_text("intelligence insights", encoding="utf-8")

    files = [hit]
    results = blog_server.search_indexes_by_parts(
        "Underdog Intelligence Network", files
    )

    assert results == [hit]


def test_locate_relevant_indexes_uses_related_terms(monkeypatch, tmp_path):
    hit = tmp_path / "hit.txt"
    hit.write_text("ally insights", encoding="utf-8")

    monkeypatch.setattr(
        blog_server, "suggest_related_terms_with_llm", lambda kw, limit=3: ["ally"]
    )

    results = blog_server.locate_relevant_indexes("friend", [hit])

    assert results == [hit]


def test_parse_related_terms_handles_multiple_formats():
    raw = "- alpha\n- beta, gamma"

    assert blog_server.parse_related_terms(raw) == ["alpha", "beta", "gamma"]


def test_parse_rahab_tracks_extracts_titles():
    html = (
        '<a href="https://rahabpunkaholicgirls.com/category/stories-en/nft-music-en/song1/">'
        "Song One</a>"
        '<a href="https://rahabpunkaholicgirls.com/category/stories-en/nft-music-en/song2/">'
        "Song Two</a>"
    )

    assert blog_server.parse_rahab_tracks(html) == ["Song One", "Song Two"]
