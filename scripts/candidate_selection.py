"""Deterministic candidate ordering with joint author limits; no model calls."""

from collections import Counter

TARGETS = {
    ("train", "experience_review"): 600,
    ("dev", "experience_review"): 60,
    ("train", "practical_notice"): 200,
    ("dev", "practical_notice"): 20,
    ("train", "opinion_column"): 400,
    ("dev", "opinion_column"): 40,
    ("train", "explanation_information"): 800,
    ("dev", "explanation_information"): 80,
}


def select_candidates(rows, evidence, used_authors, supply, limit=3418):
    """Joint author cap, deterministic deficit-balanced order; all labels remain hints."""
    bins = {}
    for row in rows:
        hint = evidence[row.source.source_doc_id]["genre_hint"]
        bins.setdefault((row.split, hint), []).append(row)
    for values in bins.values():
        values.sort(
            key=lambda r: (-evidence[r.source.source_doc_id]["score"], r.source.source_doc_id)
        )
    selected, authors, counts = [], Counter(used_authors), Counter()
    deficits = {k: max(1, n - supply.get(k[0], {}).get(k[1], 0)) for k, n in TARGETS.items()}
    while len(selected) < limit:
        for cell, values in bins.items():
            cap = 20 if cell[0] == "train" else 200
            while values and authors[(cell[0], values[0].source.author_group)] >= cap:
                values.pop(0)
        available = [k for k, v in bins.items() if v]
        if not available:
            break
        cell = min(available, key=lambda c: (counts[c] / deficits[c], c))
        row = bins[cell].pop(0)
        selected.append(row)
        counts[cell] += 1
        authors[(row.split, row.source.author_group)] += 1
    return selected, counts
