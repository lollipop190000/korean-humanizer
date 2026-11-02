"""Choose whole excerpts across an article, keeping prior frozen generation selections."""

import hashlib

from korean_humanizer.corpus_pilot import excerpt_candidates


def spread_excerpts(article, existing, reserved, provenance):
    if not existing:
        raise ValueError("coverage curation needs the original attributed source metadata")
    all_chunks = excerpt_candidates(article, provenance, max_chars=600, min_chars=200, limit=10000)
    old_by_text = {s.text: s for s in existing}
    reserved_by_text = {s.text: s for s in reserved}
    chunks = []
    template = existing[0]
    for index, chunk in enumerate(all_chunks, 1):
        if chunk.text in old_by_text:
            chunks.append(old_by_text[chunk.text])
        else:
            suffix = hashlib.sha256(chunk.text.encode()).hexdigest()[:8]
            chunks.append(
                template.model_copy(
                    update={
                        "source_doc_id": (
                            f"gyeonggi-{article['news_id']}-s600-coverage-{index:03d}-{suffix}"
                        ),
                        "text": chunk.text,
                        "provenance": provenance,
                        "rights": template.rights.model_copy(
                            update={"evidence_ref": provenance + "#kogl-1"}
                        ),
                    }
                )
            )
    if not reserved_by_text.keys() <= {s.text for s in chunks}:
        raise ValueError("a frozen selected excerpt is absent from the original article chunks")
    if len(reserved_by_text) > 4:
        raise ValueError("frozen selections exceed the document cap")
    selected = [reserved_by_text[text] for text in reserved_by_text]
    # Cover beginning, end, then interior. Reserved cases keep their exact source IDs/metadata.
    indices = [0, len(chunks) - 1, (len(chunks) - 1) // 3, 2 * (len(chunks) - 1) // 3]
    for index in indices + list(range(len(chunks))):
        if len(selected) >= 4:
            break
        if index >= 0 and chunks[index].text not in {s.text for s in selected}:
            selected.append(chunks[index])
    return selected
