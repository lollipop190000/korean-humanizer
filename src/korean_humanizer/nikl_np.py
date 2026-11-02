"""NIKL NP v1.2 SJML reader. Preserve source text; never infer usage permission."""

import hashlib
import re
import xml.etree.ElementTree as ET

from korean_humanizer.models import Rights, Source

BARE_AMP = re.compile(r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)")


def comparison_text(text):
    """Compare exporter notation without changing the stored SJML source text."""
    return "".join(re.sub(r"<name(\d+)>", r"&name\1&", text).split())


def parse_document(raw):
    text = raw.decode("utf-8-sig")
    if "<!DOCTYPE" in text or "<!ENTITY" in text:
        raise ValueError("DTD/entity declarations are not supported")
    repaired = False
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        escaped, count = BARE_AMP.subn("&amp;", text)
        if not count:
            raise
        root = ET.fromstring(escaped)
        repaired = True
    if root.tag != "SJML":
        raise ValueError("unexpected document root")
    fid = root.findtext("header/fileInfo/fileId", "")
    author = root.find("header/sourceInfo/author")
    body = root.find("text")
    if not re.fullmatch(r"WDRW\d+", fid) or author is None or body is None:
        raise ValueError("missing document metadata")
    aid = author.get("id", "")
    if not re.fullmatch(r"P\d+", aid):
        raise ValueError("missing stable author ID")
    date = body.get("date", "")
    year = int(date[:4]) if re.fullmatch(r"\d{8}", date) and date[:4] != "0000" else None
    return {
        "file_id": fid,
        "author_id": aid,
        "year": year,
        "date": date,
        "category": root.findtext("header/fileInfo/category", ""),
        "paragraphs": ["".join(p.itertext()).strip() for p in body.findall("p")],
        "literal_ampersands_escaped": repaired,
    }


def candidate_sources(doc, provenance):
    if doc["year"] is None or not 1000 <= doc["year"] < 2019:
        return [], "outside_initial_year_scope"
    if any(term in doc["category"] for term in ["시(동시)", "소설", "동화", "시나리오", "극본"]):
        return [], "literary_genre_outside_initial_scope"
    chunks, pending = [], ""
    for paragraph in doc["paragraphs"]:
        if not paragraph:
            continue
        combined = (pending + "\n" + paragraph).strip()
        if len(combined) <= 600:
            pending = combined
        else:
            if 200 <= len(pending) <= 600:
                chunks.append(pending)
            pending = paragraph if len(paragraph) <= 600 else ""
    if 200 <= len(pending) <= 600:
        chunks.append(pending)
    sources = []
    for index, chunk in enumerate(chunks[:4]):
        sha = hashlib.sha256(chunk.encode()).hexdigest()[:12]
        sources.append(
            Source(
                source_doc_id=f"nikl-np-v1.2/{doc['file_id']}/{index + 1}-{sha}",
                author_group=f"nikl-np/author/{doc['author_id']}",
                source_group=f"nikl-np/document/{doc['file_id']}",
                genre="unclassified",
                text=chunk,
                provenance=provenance,
                origin="human",
                source_year=doc["year"],
                rights=Rights(),
                protected_spans=list(dict.fromkeys(re.findall(r"&name\d+&", chunk))),
            )
        )
    return sources, "candidate" if sources else "no_complete_200_to_600_character_excerpt"
