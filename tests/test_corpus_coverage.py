"""AI-authored extraction fixtures; no actual human sources or learning examples."""

from korean_humanizer.corpus_coverage import spread_excerpts
from korean_humanizer.corpus_pilot import excerpt_candidates


def test_article_end_is_not_dropped_and_reserved_middle_excerpt_stays_identical():
    article = {
        "news_id": "201812010000000000C056",
        "published": "2018-12-01",
        "author": "가상인",
        "paragraphs": [f"가짜 시험 문단 {i}. " * 30 for i in range(10)],
    }
    old = excerpt_candidates(
        article, "fixture:not-real-rights", max_chars=600, min_chars=200, limit=4
    )
    reserved = [old[2]]
    selected = spread_excerpts(article, old, reserved, "fixture:coverage")
    assert len(selected) == 4
    assert selected[0] == reserved[0]
    assert any(article["paragraphs"][-1].strip() in source.text for source in selected)
    assert all(200 <= len(source.text) <= 600 for source in selected)
    assert all(source.rights.training == "pending" for source in selected)
