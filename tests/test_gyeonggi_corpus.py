"""Synthetic publisher markup, never human training targets or actual rights evidence."""

import pytest

from korean_humanizer.gyeonggi_corpus import article_urls, extract_candidates

URL = "https://gnews.gg.go.kr/news/news_detail.do?number=201810041553577055C109"


def page():
    prose = (
        "이 문장은 파서 검증을 위해 인공지능이 만든 가짜 문장이다. "
        "실제 체험 원문이나 학습 자료로 사용하지 않는다. "
    )
    return (
        '<meta property="og:title" content="인공지능 작성 소프트웨어 시험 자료">'
        '<div class="postinfo">작성자경기도 대학생기자단 가상인 기자 '
        'fixture@example.test 2018.10.04 15:44</div><div class="postbody">'
        '<div class="picDescription">사진 설명은 제외한다.</div><p>아쉬웠다.</p>'
        f'<p>{prose * 5}</p><p>문의 031-123-4567</p></div><div class="cc">'
        '<img src="/img/new2024/sub/img_opentype01.png" alt="공공누리">'
        '<p>위 기사는 "공공누리" '
        '<a href="https://www.kogl.or.kr/info/licenseType1.do">제1유형:출처표시</a>'
        " 조건에 따라 이용 할 수 있습니다.</p></div>"
    )


def test_explicit_license_evidence_does_not_approve_rights_or_human_review():
    article, sources = extract_candidates(page(), URL, "fixture:no-real-rights")
    assert article["author"] == "가상인"
    assert article["snapshot_revision_date_verified"] is False
    assert article["human_review_status"] == "pending"
    assert sources and "아쉬웠다." in sources[0].text
    assert all(s.rights.training == "pending" for s in sources)
    assert all(s.genre == "unclassified" for s in sources)
    assert all("사진 설명" not in s.text and "031-123" not in s.text for s in sources)
    assert all(s.source_group.startswith("gnews.gg.go.kr/") for s in sources)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("licenseType1.do", "licenseType4.do"),
        ("img_opentype01.png", "img_opentype04.png"),
        ("제1유형:출처표시", "제4유형:변경금지"),
        ("2018.10.04", "2026.10.04"),
        ("2018.10.04", "2017.10.04"),
        ("작성자경기도 대학생기자단 가상인 기자", "작성자"),
    ],
)
def test_contradictory_or_missing_license_and_date_fail_closed(old, new):
    with pytest.raises(ValueError):
        extract_candidates(page().replace(old, new), URL, "fixture:no-real-rights")


def test_ambiguous_body_and_unrelated_urls_are_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        extract_candidates(page() + '<div class="postbody">duplicate</div>', URL, "fixture")
    with pytest.raises(ValueError, match="scope"):
        extract_candidates(page(), URL.replace("gnews.gg.go.kr", "example.test"), "fixture")
    markup = "Go_Page_View_RC('201810041553577055C109','N')" * 2
    markup += "Go_Page_View_RC('202610041553577055C109','N')"
    assert article_urls(markup) == [URL]


def test_older_approved_cohort_and_unspaced_role_do_not_create_a_new_author_identity():
    markup = (
        page()
        .replace("2018.10.04", "2015.10.04")
        .replace("경기도 대학생기자단 가상인 기자", "경기도대학생기자단가상인기자")
    )
    article, _ = extract_candidates(markup, URL.replace("20181004", "20151004"), "fixture")
    assert article["author"] == "가상인"
    assert article["published"] == "2015-10-04"


def test_byline_website_does_not_hide_author_or_create_a_new_identity():
    original, _ = extract_candidates(page(), URL, "fixture")
    markup = page().replace("fixture@example.test", "https://example.test/blog/")
    article, sources = extract_candidates(markup, URL, "fixture")
    assert article["author"] == original["author"]
    assert sources[0].author_group.endswith("/가상인")
    with pytest.raises(ValueError, match="byline"):
        extract_candidates(markup.replace("경기도 대학생기자단 가상인 기자", ""), URL, "fixture")
