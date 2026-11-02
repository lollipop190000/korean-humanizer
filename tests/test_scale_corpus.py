"""Synthetic extraction fixtures; these texts are not human training data."""

from test_corpus_pilot import URL, page

from korean_humanizer.scale_corpus import article_urls, extract_candidates, genre_hint


def test_source_urls_remain_on_the_allowed_article_route_and_deduplicate():
    markup = '<a href="/news/reporterView.do?newsId=123">one</a>' * 2
    markup += '<a href="https://elsewhere.test/private">ignore</a>'
    assert article_urls(markup) == [URL]


def test_small_paragraphs_are_preserved_and_new_ids_cannot_overwrite_pilot_excerpts():
    html = page().replace('<div class="view_cont">', '<div class="view_cont"><p>아쉬웠다.</p>')
    article, sources = extract_candidates(html, URL, "fixture:not-an-actual-license")
    assert "아쉬웠다." in article["paragraphs"]
    assert any("아쉬웠다." in source.text for source in sources)
    assert all("-s600-" in source.source_doc_id for source in sources)
    assert all(
        source.genre == "unclassified" and source.rights.training == "pending" for source in sources
    )
    assert len({source.source_group for source in sources}) == 1
    assert all(200 <= len(source.text) <= 600 for source in sources)


def test_genre_hints_never_turn_into_verified_genres():
    hint = genre_hint("체험 후기", ["직접 다녀온 뒤 느꼈던 생각을 정리했다."])
    assert hint["method"] == "rules_unverified"
    assert hint["label"] == "experience_review"
    assert genre_hint("알림", ["안녕하세요."])["label"] == "unclassified"
