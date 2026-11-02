"""Synthetic SJML only, never training text or actual rights evidence."""

import xml.etree.ElementTree as ET

import pytest

from korean_humanizer.nikl_np import candidate_sources, parse_document


def fixture(body, date="20180000", genre="수필"):
    return (
        "<SJML><header><fileInfo><fileId>WDRW1900100013</fileId>"
        f"<category>비출판물 &gt; {genre}</category></fileInfo>"
        '<sourceInfo><author id="P00013">가상 필자</author></sourceInfo></header>'
        f'<text date="{date}"><p>{body}</p></text></SJML>'
    ).encode()


def test_literal_ampersands_and_redaction_markers_are_preserved():
    body = "&name1&와 A & B를 사용하는 가짜 시험 문장입니다. " * 8
    doc = parse_document(fixture(body))
    assert doc["paragraphs"] == [body.strip()]
    assert doc["literal_ampersands_escaped"] is True
    candidates, reason = candidate_sources(doc, "fixture:synthetic-only")
    assert reason == "candidate"
    assert candidates[0].protected_spans == ["&name1&"]
    assert candidates[0].rights.training == "pending"
    assert "가상 필자" not in candidates[0].model_dump_json()


def test_real_xml_entities_are_not_double_escaped_and_malformed_xml_is_rejected():
    doc = parse_document(fixture("A &amp; B &name2&"))
    assert doc["paragraphs"] == ["A & B &name2&"]
    with pytest.raises(ET.ParseError):
        parse_document(fixture("bad < unescaped markup"))
    with pytest.raises(ValueError, match="DTD"):
        parse_document(b"<!DOCTYPE SJML>" + fixture("text"))


@pytest.mark.parametrize(
    "date,genre,reason",
    [
        ("20200000", "수필", "outside_initial_year_scope"),
        ("00000000", "수필", "outside_initial_year_scope"),
        ("20180000", "소설(동화)", "literary_genre_outside_initial_scope"),
    ],
)
def test_scope_is_not_relaxed_to_fill_quotas(date, genre, reason):
    doc = parse_document(fixture("이 글은 합성 시험용입니다. " * 20, date, genre))
    assert candidate_sources(doc, "fixture")[1] == reason


def test_long_paragraph_is_not_truncated():
    doc = parse_document(fixture("가" * 601))
    assert candidate_sources(doc, "fixture")[0] == []


def test_cross_format_comparison_preserves_anonymized_identity_markers():
    from korean_humanizer.nikl_np import comparison_text

    assert comparison_text("가 <name1> 나") == comparison_text("가 &name1& 나")
    assert comparison_text("<name1>") != comparison_text("<name2>")
    assert comparison_text("가 <other> 나") != comparison_text("가 나")
