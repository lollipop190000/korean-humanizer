"""Synthetic HTML fixtures; no copied source articles."""

import json

import pytest

from korean_humanizer.corpus_pilot import excerpt_candidates, parse_article
from korean_humanizer.data import assign_splits, source_errors

URL = "https://www.korea.kr/news/reporterView.do?newsId=123"


def page(year=2018, license_type=1, author="홍길동", modified=None):
    metadata = {"datePublished": f"{year}-12-31T10:00:00+09:00"}
    if modified:
        metadata["dateModified"] = modified
    return f"""<html><head><meta property="og:title" content="소프트웨어 검사 자료">
    <script type="application/ld+json">{json.dumps(metadata)}</script></head><body>
    <p>메뉴나 최신 뉴스는 본문에 들어가면 안 된다.</p>
    <div class="info"><span>{year}.12.31</span><span>정책기자 {author}</span></div>
    <div class="view_cont"><p>{"직접 경험한 일을 설명하는 문장이다. " * 8}</p>
    <table><tr><td class="imageArea">
    <img src="do-not-fetch.jpg" alt="사용 금지 이미지 설명"></td></tr>
    <tr><td class="captions">사용 금지 사진 캡션</td></tr></table>
    <p>{"다른 내용을 설명하는 원래 문장이다. " * 8}</p><script>악성 또는 불필요한 스크립트</script>
    <p>문의: test@example.test</p><p>센터: 1670-6123</p>
    <div>정책기자단|홍길동test@example.test</div><div>원문이 아닌 작가 소개문입니다.</div></div>
    <div class="kogl_wrap">
    <img src="open_type_0{license_type}.png" alt="공공누리 {license_type}유형">
    이 자료는 텍스트에 한하여 공공누리 제{license_type}유형 조건에 따라 이용 가능합니다.</div>
    <p>관련 기사나 댓글도 본문이 아니다.</p></body></html>"""


def test_text_only_extraction_and_pending_rights():
    article = parse_article(page(), URL)
    text = "\n".join(article["paragraphs"])
    for unwanted in [
        "메뉴",
        "최신 뉴스",
        "캡션",
        "이미지 설명",
        "스크립트",
        "관련 기사",
        "test@example.test",
        "1670-6123",
        "작가 소개문",
    ]:
        assert unwanted not in text
    assert article["author"] == "홍길동"
    assert article["published"].startswith("2018")
    sources = excerpt_candidates(article, "fixture:evidence", max_chars=250)
    assert sources
    assert all("training_rights_not_approved" in source_errors(s) for s in sources)
    assert all(s.rights.training == "pending" for s in sources)
    assert len({s.source_group for s in sources}) == 1
    assert len({s.author_group for s in sources}) == 1
    assert len({r.split for r in assign_splits(sources)}) == 1


@pytest.mark.parametrize(
    "markup",
    [
        page(year=2026),
        page(license_type=2),
        page(author=""),
        page(modified="2025-01-01T00:00:00+09:00"),
        page().replace("2018.12.31", "2017.12.31"),
        page().replace('class="view_cont"', 'class="other"'),
    ],
)
def test_rejects_recent_pinned_items_wrong_rights_or_missing_evidence(markup):
    with pytest.raises(ValueError):
        parse_article(markup, URL)


def test_never_chunks_a_long_paragraph_mid_sentence():
    article = parse_article(page(), URL)
    article["paragraphs"] = ["가" * 1000, "나" * 150, "다" * 150]
    chunks = excerpt_candidates(article, "fixture", max_chars=250)
    assert len(chunks) == 2
    assert chunks[0].text == "나" * 150
    assert chunks[1].text == "다" * 150
