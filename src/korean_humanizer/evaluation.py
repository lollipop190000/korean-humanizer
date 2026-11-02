"""Blind comparison preparation and descriptive ratings; no model judge calls."""

import math
import random
from collections import Counter
from typing import Literal

from pydantic import Field

from korean_humanizer.models import Record, TextResult


class Rating(Record):
    task_id: str = Field(min_length=1)
    rater_id: str = Field(min_length=1)
    naturalness_winner: Literal["A", "B", "tie", "unratable"]
    a_major_factual_error: bool
    b_major_factual_error: bool
    a_register_drift: bool
    b_register_drift: bool
    note: str = ""


def _index(rows: list[TextResult]) -> dict[str, str]:
    result = {row.item_id: row.text for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate item_id in comparison input")
    return result


def blind_compare(
    inputs: list[TextResult],
    left: list[TextResult],
    right: list[TextResult],
    *,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    originals, a, b = _index(inputs), _index(left), _index(right)
    if not originals or originals.keys() != a.keys() or originals.keys() != b.keys():
        raise ValueError("input and candidate IDs must match exactly and be nonempty")
    rng = random.Random(seed)
    ids = sorted(originals)
    rng.shuffle(ids)
    tasks, key = [], {}
    for index, item_id in enumerate(ids):
        task_id = f"blind-{index + 1:05d}"
        swapped = bool(rng.getrandbits(1))
        tasks.append(
            {
                "task_id": task_id,
                "input": originals[item_id],
                "A": b[item_id] if swapped else a[item_id],
                "B": a[item_id] if swapped else b[item_id],
            }
        )
        key[task_id] = {
            "item_id": item_id,
            "A": "right" if swapped else "left",
            "B": "left" if swapped else "right",
        }
    return tasks, {"seed": seed, "mapping": key, "share_with_raters": False}


def summarize_ratings(ratings: list[Rating], key: dict) -> dict:
    if not ratings:
        raise ValueError("cannot summarize zero ratings")
    seen = set()
    counts: Counter = Counter()
    mappings = key.get("mapping", {})
    for row in ratings:
        identity = (row.task_id, row.rater_id)
        if identity in seen:
            raise ValueError("duplicate task/rater rating")
        seen.add(identity)
        if row.task_id not in mappings:
            raise ValueError("rating contains an unknown task_id")
        mapping = mappings[row.task_id]
        if {mapping.get("A"), mapping.get("B")} != {"left", "right"}:
            raise ValueError("invalid blinding key")
        winner = row.naturalness_winner
        counts[mapping[winner] if winner in ("A", "B") else winner] += 1
        for label, factual, drift in [
            ("A", row.a_major_factual_error, row.a_register_drift),
            ("B", row.b_major_factual_error, row.b_register_drift),
        ]:
            counts[f"{mapping[label]}_major_factual_error"] += int(factual)
            counts[f"{mapping[label]}_register_drift"] += int(drift)
    decisive = counts["left"] + counts["right"]
    # Multiple ratings of a task are correlated; do not report a naive independent CI.
    one_rating_per_task = len({row.task_id for row in ratings}) == len(ratings)
    interval = None
    if decisive and one_rating_per_task:
        p, z = counts["left"] / decisive, 1.96
        center = (p + z * z / (2 * decisive)) / (1 + z * z / decisive)
        radius = (
            z
            * math.sqrt(p * (1 - p) / decisive + z * z / (4 * decisive**2))
            / (1 + z * z / decisive)
        )
        interval = [center - radius, center + radius]
    return {
        "ratings": len(ratings),
        "rated_tasks": len({row.task_id for row in ratings}),
        "total_tasks": len(mappings),
        "counts": dict(counts),
        "left_win_rate_among_decisive": counts["left"] / decisive if decisive else None,
        "wilson_95_descriptive": interval,
        "note": (
            "Ties/unratable are separate; repeated raters need grouped analysis. "
            "No release verdict."
        ),
    }
