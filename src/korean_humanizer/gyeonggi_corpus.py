"""Gyeonggi News text candidates: explicit article KOGL-1, pre-2019, pending review."""

import hashlib
import re
from datetime import datetime

from korean_humanizer.corpus_pilot import Document, excerpt_candidates
from korean_humanizer.scale_corpus import genre_hint

HOST = "gnews.gg.go.kr"


def article_urls(markup):
    ids = re.findall(r"Go_Page_View_RC\('((?:2014|2015|2016|2017|2018)\d+C\d+)'", markup)
    return list(dict.fromkeys(f"https://{HOST}/news/news_detail.do?number={n}" for n in ids))


def extract_candidates(markup, url, provenance):
    match = re.fullmatch(
        r"https://gnews\.gg\.go\.kr/news/news_detail\.do\?number=((201[4-8])\d+C\d+)", url
    )
    if not match:
        raise ValueError("article URL outside the pre-2019 Gyeonggi scope")
    nodes = list(Document(markup).root.walk())

    def single(name):
        found = [n for n in nodes if name in n.attrs.get("class", "").split()]
        if len(found) != 1:
            raise ValueError(f"missing or ambiguous {name}")
        return found[0]

    license_node = single("cc")
    license_text = re.sub(r"\s+", " ", license_node.text()).strip()
    license_nodes = list(license_node.walk())
    if not (
        "위 기사" in license_text
        and "공공누리" in license_text
        and "제1유형:출처표시" in license_text
        and any(
            n.tag == "a"
            and re.fullmatch(
                r"https?://www\.kogl\.or\.kr/info/licenseType1\.do", n.attrs.get("href", "")
            )
            for n in license_nodes
        )
        and any(
            n.tag == "img"
            and n.attrs.get("src", "").endswith("/img_opentype01.png")
            and "공공누리" in n.attrs.get("alt", "")
            for n in license_nodes
        )
    ):
        raise ValueError("article does not explicitly grant KOGL-1")
    byline = re.sub(r"\s+", " ", single("postinfo").text()).strip()
    date = re.search(r"\b(201[4-8]\.\d{2}\.\d{2})\s+\d{2}:\d{2}\s*$", byline)
    if not date or date[1][:4] != match[2]:
        raise ValueError("visible date missing or disagrees with article identifier")
    published = datetime.strptime(date[1], "%Y.%m.%d").date().isoformat()
    author = re.sub(r"https?://\S+|\S+@\S+", "", byline[: date.start()])
    author = author.removeprefix("작성자").strip()
    if not re.fullmatch(r"[가-힣A-Za-z· ]{2,60}", author):
        raise ValueError("named byline is missing or ambiguous")
    author_credit = author
    person = re.fullmatch(r".*기자단\s*([가-힣]{2,5}?)(?:\s*기자)?", author)
    if person:
        author = person[1]
    body = single("postbody")
    # The publisher mixes photo captions into prose, outside figure elements.
    for node in body.walk():
        if "picDescription" in node.attrs.get("class", "").split():
            node.children = []
    paragraphs = []
    for line in body.text().splitlines():
        text = re.sub(r"\s+", " ", line).strip()
        if len(text) < 2 or not re.search(r"[가-힣]", text):
            continue
        if re.search(r"@|ⓒ|https?://|\b0\d{1,2}[- ]\d{3,4}[- ]\d{4}\b", text):
            continue
        paragraphs.append(text)
    if not paragraphs:
        raise ValueError("no prose paragraphs remained")
    titles = [
        n.attrs.get("content", "")
        for n in nodes
        if n.tag == "meta" and n.attrs.get("property") == "og:title"
    ]
    if len(titles) != 1 or not titles[0]:
        raise ValueError("missing or ambiguous article title")
    article = {
        "url": url,
        "news_id": match[1],
        "title": titles[0],
        "author": author,
        "author_credit": author_credit,
        "published": published,
        "modified": None,
        "license": "KOGL-1 article; images excluded",
        "license_excerpt": license_text,
        "paragraphs": paragraphs,
        "html_sha256": hashlib.sha256(markup.encode()).hexdigest(),
        "human_review_status": "pending",
        "snapshot_revision_date_verified": False,
        "genre_hint": genre_hint(titles[0], paragraphs),
        "extraction_policy": "short_paragraphs_preserved_s600_min200_doccap4",
    }
    sources = []
    for index, source in enumerate(
        excerpt_candidates(article, provenance, min_chars=200, max_chars=600, limit=4), 1
    ):
        suffix = hashlib.sha256(source.text.encode()).hexdigest()[:8]
        sources.append(
            source.model_copy(
                update={
                    "source_doc_id": f"gyeonggi-{match[1]}-s600-{index:03d}-{suffix}",
                    "source_group": f"gnews.gg.go.kr/article/{match[1]}",
                    "author_group": f"gnews.gg.go.kr/byline/{author}",
                    "genre": "unclassified",
                }
            )
        )
    article["excerpt_ids"] = [s.source_doc_id for s in sources]
    return article, sources
