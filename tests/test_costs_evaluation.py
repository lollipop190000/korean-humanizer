from decimal import Decimal

import pytest
from pydantic import ValidationError

from korean_humanizer.costs import CostConfig, estimate
from korean_humanizer.evaluation import Rating, blind_compare, summarize_ratings
from korean_humanizer.models import TextResult


def test_estimate_accounts_for_judging_gpu_storage_and_budget():
    config = CostConfig(
        pricing_status="test",
        budget_usd="5",
        gpu_hourly_usd="2",
        gpu_max_hours="2",
        storage_gb="30",
        storage_days="30",
        storage_usd_per_gb_month="0.1",
        contingency_fraction="0",
        api=[
            {
                "name": "judge",
                "requests": 100,
                "input_tokens_per_request": 1000,
                "max_output_tokens_per_request": 500,
                "input_usd_per_million": "1",
                "output_usd_per_million": "2",
            }
        ],
    )
    result = estimate(config)
    assert Decimal(result["total_with_contingency_usd"]) == Decimal("7.2")
    assert not result["within_budget"]
    assert not result["execution_enabled"]


@pytest.mark.parametrize("price", ["-1", "NaN", "Infinity"])
def test_invalid_prices_rejected(price):
    with pytest.raises(ValidationError):
        CostConfig(pricing_status="test", budget_usd="1", gpu_hourly_usd=price)


def test_blinding_retains_alignment_but_hides_candidate_identity():
    rows = [TextResult(item_id=str(i), text=f"입력 {i}") for i in range(12)]
    left = [row.model_copy(update={"text": "왼쪽 " + row.item_id}) for row in rows]
    right = [row.model_copy(update={"text": "오른쪽 " + row.item_id}) for row in rows]
    tasks, key = blind_compare(rows, left[::-1], right)
    assert {mapping["A"] for mapping in key["mapping"].values()} == {"left", "right"}
    for task in tasks:
        mapping = key["mapping"][task["task_id"]]
        assert set(task) == {"task_id", "input", "A", "B"}
        assert task["A"] == ("왼쪽 " if mapping["A"] == "left" else "오른쪽 ") + mapping["item_id"]
    with pytest.raises(ValueError, match="match exactly"):
        blind_compare(rows, left[:-1], right)


def test_ratings_keep_ties_and_error_counts_separate():
    key = {"mapping": {"x": {"A": "right", "B": "left"}}}
    rating = Rating(
        task_id="x",
        rater_id="r1",
        naturalness_winner="tie",
        a_major_factual_error=True,
        b_major_factual_error=False,
        a_register_drift=False,
        b_register_drift=False,
    )
    result = summarize_ratings([rating], key)
    assert result["counts"]["tie"] == 1
    assert result["counts"]["right_major_factual_error"] == 1
    assert result["left_win_rate_among_decisive"] is None
    with pytest.raises(ValueError, match="duplicate"):
        summarize_ratings([rating, rating], key)
