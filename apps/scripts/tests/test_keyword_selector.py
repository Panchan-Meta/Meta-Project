from apps.scripts.keyword_selector import generate_search_query, select_keywords


def test_generate_search_query_is_pure():
    assert generate_search_query("demo") == "demo insights"
    assert generate_search_query("  ") == ""


def test_select_keywords_reproducible_with_seed():
    keywords = ["a", "b", "c"]

    first = select_keywords(keywords, seed=42, limit=2)
    second = select_keywords(keywords, seed=42, limit=2)

    assert first == second
    assert len(first) == 2
