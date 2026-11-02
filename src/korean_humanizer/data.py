"""Source policy and deterministic grouping before draft generation."""

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from korean_humanizer.models import AssignedSource, Source


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).casefold())


def source_errors(
    source: Source,
    *,
    allow_fixture: bool = False,
    external_transfer: bool = False,
    allow_later_years: bool = False,
) -> list[str]:
    errors = []
    fixture = source.origin == "synthetic_fixture"
    if fixture and not allow_fixture:
        errors.append("synthetic_fixture_not_human_training_data")
    if source.rights.training != "approved":
        errors.append("training_rights_not_approved")
    if source.rights.derivative_creation != "approved":
        errors.append("derivative_rights_not_approved")
    if not source.rights.evidence_ref:
        errors.append("missing_rights_evidence")
    if external_transfer and source.rights.third_party_transfer != "approved":
        errors.append("third_party_transfer_not_approved")
    if not fixture:
        if source.source_year is None:
            errors.append("unverified_authorship_year")
        elif source.source_year >= 2019 and not allow_later_years:
            errors.append("outside_initial_pre_2019_policy")
    if any(span not in source.text for span in source.protected_spans):
        errors.append("protected_span_not_in_source")
    return errors


def _shingles(text: str) -> set[str]:
    return {text[index : index + 5] for index in range(len(text) - 4)}


def assign_splits(
    sources: list[Source],
    *,
    seed: int = 42,
    dev_fraction: float = 0.15,
    test_fraction: float = 0.15,
    near_duplicate_threshold: float = 0.85,
) -> list[AssignedSource]:
    if not sources:
        raise ValueError("cannot split an empty dataset")
    if not (0 < dev_fraction < 1 and 0 < test_fraction < 1 and dev_fraction + test_fraction < 1):
        raise ValueError("split fractions must leave nonempty train/dev/test probabilities")
    if not 0 < near_duplicate_threshold <= 1:
        raise ValueError("near duplicate threshold must be in (0, 1]")
    ids = [source.source_doc_id for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "source_doc_id must identify a unique record; use source_group for excerpts"
        )
    parent = list(range(len(sources)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        parent[find(left)] = find(right)

    seen: dict[tuple[str, str], int] = {}
    texts = [normalized(source.text) for source in sources]
    for index, source in enumerate(sources):
        keys = [
            ("author", source.author_group),
            ("source", source.source_group),
            ("text", texts[index]),
        ]
        for key in keys:
            if key in seen:
                union(index, seen[key])
            seen[key] = index

    # CPU pilot implementation. Exact/near duplicates and shared identities are transitive.
    shingles = [_shingles(text) if len(text) >= 30 else set() for text in texts]
    character_counts = [Counter(text) for text in texts]
    # Bucket overlap is an upper bound on exact character overlap. It can reject
    # impossible close-edit matches cheaply, but never establish a match itself.
    character_buckets = []
    for counts in character_counts:
        buckets = [0] * 16
        for char, count in counts.items():
            buckets[ord(char) % 16] += count
        character_buckets.append(buckets)
    postings: dict[str, set[int]] = defaultdict(set)
    for index, current in enumerate(shingles):
        candidates: Counter[int] = Counter()
        for shingle in current:
            candidates.update(postings[shingle])
        for other, common in candidates.items():
            if find(index) == find(other):
                continue
            union_size = len(current) + len(shingles[other]) - common
            jaccard = common / union_size if union_size else 0
            # A one-syllable change in a short Korean paragraph disrupts several shingles.
            close_edit = False
            if jaccard < near_duplicate_threshold:
                total = len(texts[index]) + len(texts[other])
                if total and 2 * min(len(texts[index]), len(texts[other])) / total >= 0.95:
                    bucket_overlap = sum(
                        min(left, right)
                        for left, right in zip(
                            character_buckets[index], character_buckets[other], strict=True
                        )
                    )
                    if 2 * bucket_overlap / total < 0.95:
                        continue
                    left, right = character_counts[index], character_counts[other]
                    if len(left) > len(right):
                        left, right = right, left
                    matched = sum(min(count, right.get(char, 0)) for char, count in left.items())
                    # Same multiset upper bound as SequenceMatcher.quick_ratio,
                    # reusing per-document counts instead of rebuilding per pair.
                    if 2 * matched / total >= 0.95:
                        close_edit = (
                            SequenceMatcher(
                                None, texts[index], texts[other], autojunk=False
                            ).ratio()
                            >= 0.95
                        )
            if jaccard >= near_duplicate_threshold or close_edit:
                union(index, other)
        for shingle in current:
            postings[shingle].add(index)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(sources)):
        groups[find(index)].append(index)
    assignments = {}
    for members in groups.values():
        key = "\n".join(sorted(ids[index] for index in members))
        group_id = hashlib.sha256(key.encode()).hexdigest()
        score = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest(), 16) / 2**256
        split = (
            "test"
            if score < test_fraction
            else ("dev" if score < test_fraction + dev_fraction else "train")
        )
        for index in members:
            assignments[index] = (split, group_id)
    return [
        AssignedSource(source=source, split=assignments[index][0], group_id=assignments[index][1])
        for index, source in enumerate(sources)
    ]


def extend_frozen_splits(
    frozen: list[AssignedSource], additions: list[Source], *, seed: int = 42
) -> tuple[list[AssignedSource], list[Source]]:
    """Preserve old rows; quarantine new components bridging different frozen splits."""
    if not frozen:
        raise ValueError("extension requires an existing frozen split")
    existing = {row.source.source_doc_id: row for row in frozen}
    if len(existing) != len(frozen):
        raise ValueError("duplicate frozen source IDs")
    new = {}
    for source in additions:
        identifier = source.source_doc_id
        if identifier in existing:
            if source != existing[identifier].source:
                raise ValueError("conflicting metadata for an existing frozen source")
            continue
        if identifier in new:
            raise ValueError("duplicate added source IDs")
        new[identifier] = source
    grouped = assign_splits([row.source for row in frozen] + list(new.values()), seed=seed)
    components = defaultdict(list)
    for row in grouped:
        components[row.group_id].append(row)
    added = []
    quarantined = []
    for group_id, component in components.items():
        old = [
            existing[r.source.source_doc_id]
            for r in component
            if r.source.source_doc_id in existing
        ]
        newcomers = [r for r in component if r.source.source_doc_id in new]
        previous_splits = {r.split for r in old}
        if len(previous_splits) > 1:
            if not newcomers:
                raise ValueError("the original frozen manifest already leaks related sources")
            quarantined.extend(r.source for r in newcomers)
            continue
        split = next(iter(previous_splits)) if old else component[0].split
        old_groups = {r.group_id for r in old}
        preserved_group = next(iter(old_groups)) if len(old_groups) == 1 else group_id
        added.extend(
            AssignedSource(source=r.source, split=split, group_id=preserved_group)
            for r in newcomers
        )
    return frozen + sorted(added, key=lambda r: r.source.source_doc_id), sorted(
        quarantined, key=lambda source: source.source_doc_id
    )
