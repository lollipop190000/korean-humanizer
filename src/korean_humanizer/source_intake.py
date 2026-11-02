"""Offline intake of normalized sources; never grants rights or starts generation."""

from collections import Counter

from korean_humanizer.data import extend_frozen_splits, normalized
from korean_humanizer.source_scope import exclusion_reason


def prepare_intake(frozen, candidates):
    """Return the preserved base, new split rows, and reasons for excluded candidates.

    Author/document groups must already be canonical provenance identifiers. Native
    corpus conversion and cross-publisher identity mapping are separate steps.
    """
    if not frozen:
        raise ValueError("an existing frozen inventory is required")
    prior = {r.source.source_doc_id: r.source for r in frozen}
    if len(prior) != len(frozen):
        raise ValueError("duplicate frozen source IDs")
    identifiers = [s.source_doc_id for s in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate candidate source IDs")
    seen_text = {normalized(s.text) for s in prior.values()}
    eligible, excluded, reused = [], [], []
    for source in sorted(candidates, key=lambda s: s.source_doc_id):
        identifier = source.source_doc_id
        if identifier in prior:
            if source != prior[identifier]:
                raise ValueError("conflicting metadata for an existing frozen source")
            reused.append(identifier)
            continue
        reason = ""
        if source.origin != "human":
            reason = "not_human_origin"
        elif source.source_year is None:
            reason = "missing_document_year"
        elif source.source_year >= 2019:
            reason = "outside_pre_2019_scope"
        elif not 200 <= len(source.text) <= 600:
            reason = "outside_200_to_600_characters"
        else:
            reason = exclusion_reason(source.text)
        if reason:
            excluded.append({"source_doc_id": identifier, "reason": reason})
            continue
        if normalized(source.text) in seen_text:
            # Keep identity links during grouping: dropping a duplicate first
            # could separate its other author/document records from a holdout.
            excluded.append({"source_doc_id": identifier, "reason": "normalized_duplicate"})
        seen_text.add(normalized(source.text))
        eligible.append(source)
    extended, conflicts = extend_frozen_splits(frozen, eligible)
    excluded.extend(
        {"source_doc_id": s.source_doc_id, "reason": "conflicting_frozen_splits"} for s in conflicts
    )
    excluded_ids = {r["source_doc_id"] for r in excluded}
    additions = [r for r in extended[len(frozen) :] if r.source.source_doc_id not in excluded_ids]
    report = {
        "status": "prepared_pending_rights_genre_and_semantic_review",
        "input_candidates": len(candidates),
        "reused_frozen_sources": len(reused),
        "frozen_rows": len(frozen),
        "frozen_rows_unchanged": extended[: len(frozen)] == frozen,
        "new_sources": len(additions),
        "linkage_only_rows": len(extended) - len(frozen) - len(additions),
        "new_sources_by_split": dict(Counter(r.split for r in additions)),
        "excluded_by_reason": dict(Counter(r["reason"] for r in excluded)),
        "rights_approved_by_intake": 0,
        "genre_quotas_verified": False,
        "generation_executed": False,
        "training_executed": False,
        "external_spend_usd": 0,
    }
    return extended, excluded, report
