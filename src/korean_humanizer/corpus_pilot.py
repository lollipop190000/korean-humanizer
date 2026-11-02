"""Extract a small KOGL-1 text pilot, preserving evidence and pending human review."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser

from korean_humanizer.models import Rights, Source

VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
EXCLUDE = {
    "script",
    "style",
    "table",
    "figure",
    "figcaption",
    "img",
    "iframe",
    "video",
    "audio",
    "object",
    "button",
    "form",
}
BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "blockquote"}


@dataclass
class Element:
    tag: str
    attrs: dict = field(default_factory=dict)
    children: list = field(default_factory=list)

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, Element):
                yield from child.walk()

    def text(self, clean=True):
        classes = set(self.attrs.get("class", "").split())
        if clean and (
            self.tag in EXCLUDE or classes & {"captions", "imageArea", "source", "reporter_profile"}
        ):
            return ""
        value = "".join(c.text(clean) if isinstance(c, Element) else c for c in self.children)
        return "\n" + value + "\n" if self.tag in BLOCK else value


class Document(HTMLParser):
    def __init__(self, markup):
        super().__init__(convert_charrefs=True)
        self.root = Element("root")
        self.stack = [self.root]
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        node = Element(tag, dict(attrs))
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                self.stack = self.stack[:i]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def parse_article(markup: str, url: str, *, min_paragraph_chars: int = 30):
    if not re.fullmatch(r"https://www\.korea\.kr/news/reporterView\.do\?newsId=\d+", url):
        raise ValueError("only explicit reporter article URLs are supported")
    nodes = list(Document(markup).root.walk())

    def byclass(name):
        return [n for n in nodes if name in n.attrs.get("class", "").split()]

    body = byclass("view_cont")
    license_nodes = byclass("kogl_wrap")
    if len(body) != 1 or len(license_nodes) != 1:
        raise ValueError("missing or ambiguous article/license containers")
    license_text = re.sub(r"\s+", " ", license_nodes[0].text()).strip()
    images = [n.attrs for n in license_nodes[0].walk() if n.tag == "img"]
    if not (
        "텍스트에 한하여" in license_text
        and "제1유형" in license_text
        and any("open_type_01" in a.get("src", "") and "1유형" in a.get("alt", "") for a in images)
    ):
        raise ValueError("article does not explicitly grant KOGL-1 for text")
    articles = []
    for n in nodes:
        if n.tag == "script" and n.attrs.get("type") == "application/ld+json":
            try:
                value = json.loads(n.text(clean=False))
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "datePublished" in value:
                articles.append(value)
    if len(articles) != 1:
        raise ValueError("publication date is missing or ambiguous")
    article = articles[0]
    date = datetime.fromisoformat(article["datePublished"].replace("Z", "+00:00"))
    if date.year >= 2019:
        raise ValueError("article is outside the pre-2019 pilot")
    modified = article.get("dateModified")
    if modified and datetime.fromisoformat(modified.replace("Z", "+00:00")).year >= 2019:
        raise ValueError("article was modified after the initial date cutoff")
    bylines = []
    for info in byclass("info"):
        text = re.sub(r"\s+", " ", info.text()).strip()
        text = re.sub(r"^(\d{4}\.\d{2}\.\d{2})(?=\S)", r"\1 ", text)
        match = re.fullmatch(r"(\d{4}\.\d{2}\.\d{2}) 정책기자\s+([가-힣A-Za-z· ]{2,40})", text)
        if match:
            if match[1] != date.strftime("%Y.%m.%d"):
                raise ValueError("visible and structured publication dates disagree")
            bylines.append(match[2].strip())
    if len(bylines) != 1:
        raise ValueError("a named reporter byline is required")
    paragraphs = []
    body_text = re.split(r"(?m)^\s*정책기자단\s*\|", body[0].text(), maxsplit=1)[0]
    for line in body_text.splitlines():
        text = re.sub(r"\s+", " ", line).strip()
        # Embedded contact details and boilerplate cannot become training targets.
        if not text or re.search(
            r"정책기자단\s+[가-힣]{2,5}\s*\(|@|"
            r"\b(?:0\d{1,2}[- ]\d{3,4}[- ]\d{4}|1[568]\d{2}[- ]\d{4})\b",
            text,
        ):
            continue
        if text.startswith(("관련기사", "관련 기사", "대한민국 정책기자단 ")) and len(text) < 100:
            continue
        if len(text) >= min_paragraph_chars and re.search("[가-힣]", text):
            paragraphs.append(text)
    if not paragraphs:
        raise ValueError("no prose paragraphs remained")
    title = next(
        (
            n.attrs["content"]
            for n in nodes
            if n.tag == "meta" and n.attrs.get("property") == "og:title"
        ),
        "",
    )
    return {
        "url": url,
        "news_id": url.split("newsId=")[-1],
        "title": title,
        "author": bylines[0],
        "published": article["datePublished"],
        "modified": modified,
        "license": "KOGL-1 text only",
        "license_excerpt": license_text[:500],
        "paragraphs": paragraphs,
        "html_sha256": hashlib.sha256(markup.encode()).hexdigest(),
        "human_review_status": "pending",
        "snapshot_revision_date_verified": bool(modified),
    }


def excerpt_candidates(article: dict, provenance: str, *, max_chars=900, min_chars=120, limit=12):
    chunks = []
    current = []
    for paragraph in article["paragraphs"]:
        if len(paragraph) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
            continue
        if current and len("\n\n".join(current + [paragraph])) > max_chars:
            chunks.append("\n\n".join(current))
            current = []
        current.append(paragraph)
    if current:
        chunks.append("\n\n".join(current))
    chunks = [text for text in chunks if len(text) >= min_chars][:limit]
    return [
        Source(
            source_doc_id=f"korea-{article['news_id']}-{i:03d}",
            author_group="korea.kr/policy-reporter/" + article["author"],
            source_group="korea.kr/article/" + article["news_id"],
            genre="public_policy_prose",
            text=text,
            provenance=provenance,
            origin="human",
            source_year=int(article["published"][:4]),
            # Attribution and exact extracted text/date remain available for human verification.
            # A machine finding a license badge is not recorded as a person's approval.
            rights=Rights(evidence_ref=provenance + "#kogl-1"),
        )
        for i, text in enumerate(chunks, 1)
    ]
