"""Standalone, standard-library-only local selector. No network, GPU or rights grant."""

import argparse
import hashlib
import io
import itertools
import json
import random
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

TARGETS = ("NIKL_NEWSPAPER_v2.0_JSON.zip", "NIKL_WRITTEN_v1.2_JSON.zip")
QUOTAS = {
    "experience_review": 1200,
    "explanation_information": 1600,
    "opinion_column": 3000,
    "practical_notice": 2200,
}
FOCUSED_QUOTAS = {"experience_review": 2500, "opinion_column": 2000, "practical_notice": 2000}
# The Windows builder can replace this with hashes of already received texts.
EXCLUDED_TEXT_HASHES = set()
EXISTING_DOCUMENT_COUNTS = {}
PATTERNS = {
    "experience_review": r"다녀|느꼈|내가|나에게|경험했|체험했|방문했|돌아보니",
    "opinion_column": (
        r"생각한다|생각합니다|필요하다|필요합니다|해야 한다|해야한다|"
        r"문제점|바란다|바랍니다|아쉽"
    ),
    "practical_notice": (
        r"신청|접수|제출|마감|모집|준비물|이용 방법|이용방법|"
        r"참가 방법|참가방법|신청 자격|신청자격"
    ),
}


class JSONStream:
    def __init__(self, stream):
        self.stream, self.buffer, self.eof = stream, "", False
        self.decoder = json.JSONDecoder()

    def more(self):
        chunk = self.stream.read(65536)
        self.eof = not chunk
        self.buffer += chunk
        if len(self.buffer) > 16 * 1024 * 1024:
            raise ValueError("document_or_header_exceeds_16M_characters")

    def whitespace(self):
        while True:
            self.buffer = self.buffer.lstrip()
            if self.buffer or self.eof:
                return
            self.more()

    def consume(self, char):
        self.whitespace()
        if not self.buffer.startswith(char):
            raise ValueError("unexpected_or_truncated_JSON")
        self.buffer = self.buffer[len(char) :]

    def value(self):
        self.whitespace()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer)
                self.buffer = self.buffer[end:]
                return value
            except json.JSONDecodeError:
                if self.eof:
                    raise ValueError("truncated_or_invalid_JSON") from None
                self.more()


def documents(stream):
    """Yield complete document objects only; root year is never an authorship date."""
    reader = JSONStream(stream)
    reader.consume("{")
    header = {}
    while True:
        key = reader.value()
        reader.consume(":")
        if key == "document":
            if not isinstance(header.get("metadata"), dict):
                raise ValueError("metadata_must_precede_document_array")
            reader.consume("[")
            reader.whitespace()
            if reader.buffer.startswith("]"):
                reader.consume("]")
            else:
                while True:
                    doc = reader.value()
                    if not isinstance(doc, dict):
                        raise ValueError("document_must_be_object")
                    yield header, doc
                    reader.whitespace()
                    if reader.buffer.startswith("]"):
                        reader.consume("]")
                        break
                    reader.consume(",")
            reader.consume("}")
            reader.whitespace()
            if reader.buffer:
                raise ValueError("unexpected_trailing_content")
            return
        header[key] = reader.value()
        reader.consume(",")


def normalized(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def triage(text, topic):
    scores = {k: len(re.findall(v, text)) for k, v in PATTERNS.items()}
    if re.search(r"사설|칼럼|기고|논단|시론|오피니언", topic) and scores["opinion_column"]:
        scores["opinion_column"] += 3
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "explanation_information", scores


def focused_triage(text, topic):
    """Require positive purpose cues; these remain hints, never validated genres."""
    unquoted = re.sub(r'["“「『][^"”」』]*["”」』]', " ", text)
    reporting = len(re.findall(r"고\s?(?:밝혔|말했|전했|설명했)", unquoted))
    directions = len(
        re.findall(
            r"신청\s?방법|접수\s?방법|이용\s?방법|제출\s?서류|신청\s?자격|"
            r"참가\s?자격|접수\s?기간|신청\s?기간|유의\s?사항|준비물|"
            r"신청하려|참여하려|희망하는|원하는\s?(?:사람|분|시민|학생)|"
            r"하세요|하십시오|주시기|바랍니다",
            unquoted,
        )
    )
    actions = len(re.findall(r"신청|접수|제출|마감|모집|예약|이용|수강|참가", unquoted))
    deadline = bool(re.search(r"(?:\d+일|\d+시|요일).{0,8}(?:까지|부터)", unquoted))
    scores = {k: 0 for k in FOCUSED_QUOTAS}
    if directions and actions >= 2 and (directions >= 2 or deadline):
        scores["practical_notice"] = 4 * directions + actions + 2 * deadline - reporting
    first_person = len(
        re.findall(r"(?:^|\s)(?:나는|내가|나에게|나의|내게|저는|제가|나를)\s", unquoted)
    )
    experience = len(
        re.findall(r"느꼈|겪었|다녀|돌아보니|깨달았|기억|경험|처음|어릴|살았|보았다", unquoted)
    )
    if first_person and experience and not reporting:
        scores["experience_review"] = 4 * first_person + 2 * experience
    editorial = bool(re.search(r"사설|칼럼|기고|논단|시론|오피니언", topic))
    argument = len(
        re.findall(
            r"생각한다|생각합니다|해야 한다|필요하다|그러므로|따라서|그러나|문제는", unquoted
        )
    )
    if argument >= 2 and (editorial or not reporting):
        scores["opinion_column"] = 6 * editorial + 2 * argument - reporting
    best = max(scores, key=scores.get)
    return (best if scores[best] > 0 else None), scores


def candidates(header, doc, *, focused=False, max_excerpts=4):
    meta = doc.get("metadata", {})
    date = str(meta.get("date", ""))
    if not re.fullmatch(r"\d{8}", date) or not 1000 <= int(date[:4]) < 2019:
        return []
    category = str(header.get("metadata", {}).get("category", ""))
    if re.search(r"책-상상|문학|소설|동화|시집|희곡", category):
        return []
    author = meta.get("author", "")
    if not isinstance(author, str) or normalized(author) in {
        "",
        "미상",
        "없음",
        "알수없음",
        "null",
    }:
        return []
    author_id = hashlib.sha256(normalized(author).encode()).hexdigest()
    document_id = doc.get("id")
    if not isinstance(document_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", document_id):
        return []
    paragraphs = doc.get("paragraph", [])
    if not isinstance(paragraphs, list):
        return []
    topic = " ".join(str(meta.get(k, "")) for k in ["topic", "original_topic", "title"])
    if paragraphs and isinstance(paragraphs[0].get("form"), str):
        topic += " " + paragraphs[0]["form"][:150]
    chunks, pending, ids = [], "", []
    for p in paragraphs:
        text = p.get("form")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        joined = (pending + "\n" + text).strip()
        if len(joined) <= 600:
            pending, ids = joined, ids + [p.get("id", "")]
        else:
            if 200 <= len(pending) <= 600:
                chunks.append((pending, ids))
            pending, ids = (text, [p.get("id", "")]) if len(text) <= 600 else ("", [])
    if 200 <= len(pending) <= 600:
        chunks.append((pending, ids))
    results = []
    for text, paragraph_ids in chunks:
        if re.search(
            r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b0\d{1,2}[- ]\d{3,4}[- ]\d{4}\b|"
            r"<\|im_start\|>|복용량|투약|법률\s*자문",
            text,
        ):
            continue
        genre, scores = focused_triage(text, topic) if focused else triage(text, topic)
        if genre is None:
            continue
        results.append(
            {
                "document_id": document_id,
                "author_group_hash": author_id,
                "document_date": date,
                "category": category,
                "publisher": str(meta.get("publisher", "")),
                "paragraph_ids": paragraph_ids,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "genre_hint": genre,
                "hint_scores": scores,
                "rights_status": "pending_agreement_review",
                "human_review_status": "pending",
            }
        )
    # Prefer deficient genres within a long document, without rewriting its content.
    return sorted(
        results,
        key=lambda r: (
            False if focused else r["genre_hint"] not in {"opinion_column", "practical_notice"},
            -sum(r["hint_scores"].values()),
            r["text_sha256"],
        ),
    )[:max_excerpts]


def write_parts(rows, output, summary, byte_limit=16 * 1024 * 1024):
    parts, current, size = [], [], 0
    for row in rows:
        encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode()
        if len(encoded) > byte_limit:
            raise ValueError("individual_record_exceeds_part_limit")
        if current and size + len(encoded) > byte_limit:
            parts.append(b"".join(current))
            current, size = [], 0
        current.append(encoded)
        size += len(encoded)
    if current:
        parts.append(b"".join(current))
    # Even an empty selection returns a diagnostic package, never a fake success.
    for i, data in enumerate(parts or [b""], 1):
        path = output / f"NIKL-selected-{i:03d}.zip"
        manifest = dict(
            summary,
            part=i,
            parts=max(1, len(parts)),
            rows_in_part=data.count(b"\n"),
            content_sha256=hashlib.sha256(data).hexdigest(),
        )
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            z.writestr("candidates.private.jsonl", data)
        print(f"Created: {path.name} ({path.stat().st_size / 1024**2:.2f} MB)", flush=True)


def select(root, output, max_documents=80000, max_members=512, per_member=250, *, focused=False):
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for name in TARGETS:
        matches = list(root.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"Expected one {name}; found {len(matches)}")
        files.append(matches[0])
    counts, authors, docs, seen = Counter(), Counter(), Counter(EXISTING_DOCUMENT_COUNTS), set()
    rows, errors, source_info = [], [], []
    scanned = 0
    quotas = FOCUSED_QUOTAS if focused else QUOTAS
    prior_duplicates = 0
    archives = [zipfile.ZipFile(path) for path in files]
    try:
        lists = []
        for path, archive in zip(files, archives, strict=True):
            entries = sorted(
                (e for e in archive.infolist() if e.filename.endswith(".json")),
                key=lambda e: e.filename,
            )
            random.Random(42).shuffle(entries)
            lists.append(entries[:max_members])
            source_info.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "json_members": len(entries),
                    "scheduled_members": min(len(entries), max_members),
                }
            )
        for round_entries in itertools.zip_longest(*lists):
            for index, entry in enumerate(round_entries):
                if entry is None or scanned >= max_documents:
                    continue
                print(
                    f"Scanning {files[index].name}: {Path(entry.filename).name}; "
                    f"documents={scanned}, selected={len(rows)}",
                    flush=True,
                )
                try:
                    with (
                        archives[index].open(entry) as binary,
                        io.TextIOWrapper(binary, encoding="utf-8-sig") as stream,
                    ):
                        for header, doc in itertools.islice(
                            documents(stream), min(per_member, max_documents - scanned)
                        ):
                            scanned += 1
                            for row in candidates(header, doc, focused=focused):
                                genre, author = row["genre_hint"], row["author_group_hash"]
                                key = normalized(row["text"])
                                if hashlib.sha256(key.encode()).hexdigest() in EXCLUDED_TEXT_HASHES:
                                    prior_duplicates += 1
                                    continue
                                doc_id = row["document_id"]
                                if (
                                    counts[genre] >= quotas[genre]
                                    or authors[author] >= 20
                                    or docs[doc_id] >= 4
                                    or key in seen
                                ):
                                    continue
                                row.update(
                                    source_zip=files[index].name,
                                    source_member=entry.filename,
                                    source_member_crc32=f"{entry.CRC:08x}",
                                    source_member_bytes=entry.file_size,
                                )
                                rows.append(row)
                                counts[genre] += 1
                                authors[author] += 1
                                docs[doc_id] += 1
                                seen.add(key)
                except (
                    ValueError,
                    UnicodeError,
                    zipfile.BadZipFile,
                    KeyError,
                    TypeError,
                    AttributeError,
                ) as exc:
                    errors.append(
                        {
                            "archive": files[index].name,
                            "member": entry.filename,
                            "error_type": type(exc).__name__,
                        }
                    )
                if scanned >= max_documents or all(counts[k] >= v for k, v in quotas.items()):
                    break
            if scanned >= max_documents or all(counts[k] >= v for k, v in quotas.items()):
                break
    finally:
        for archive in archives:
            archive.close()
    summary = {
        "schema": "nikl-local-selection-v1",
        "scanned_complete_documents": scanned,
        "selected_excerpts": len(rows),
        "focused_purpose_cues": focused,
        "previously_received_texts_excluded": prior_duplicates,
        "unverified_genre_hints": dict(counts),
        "author_groups": len(authors),
        "source_archives": source_info,
        "limits": {
            "documents": max_documents,
            "members_per_archive": max_members,
            "documents_per_member": per_member,
            "candidates_by_hint": quotas,
        },
        "sampling_note": (
            "Bounded prefix scan of seeded shuffled members; "
            "not representative sampling or proof of full corpus supply."
        ),
        "errors": errors,
        "rights_approved": False,
        "training_executed": False,
        "network_uploads": 0,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_parts(rows, output, summary)
    print(
        f"DONE. Selected {len(rows)} candidate excerpts. Upload only NIKL-selected-*.zip.",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focused", action="store_true")
    args = parser.parse_args()
    if args.focused:
        select(args.root, args.output, 800000, 12000, 3000, focused=True)
    else:
        select(args.root, args.output)
