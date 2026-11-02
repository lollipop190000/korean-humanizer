"""Pure fake phase reservations; these tests do not authorize or create paid resources."""

import pytest

from korean_humanizer.scale_budget import empty_ledger, finish_phase, reserve_phase, used_seconds


def test_aggregate_reservation_and_no_second_gpu_until_confirmed_deleted():
    ledger = empty_ledger("fixture:not-real-user-approval")
    reserved = reserve_phase(ledger, "a" * 64, "drafts", ["a"], 3600)
    assert ledger["phases"] == []
    with pytest.raises(ValueError, match="unconfirmed"):
        reserve_phase(reserved, "b" * 64, "judge", ["a"], 3600)
    with pytest.raises(ValueError, match="confirmed termination"):
        finish_phase(reserved, "a" * 64, 600, termination_confirmed=False)
    ended = finish_phase(reserved, "a" * 64, 600, termination_confirmed=True)
    assert used_seconds(ended) == 600
    assert reserve_phase(ended, "b" * 64, "judge", ["a"], 3600)
    with pytest.raises(ValueError, match="rerun"):
        reserve_phase(ended, "a" * 64, "drafts", ["another"], 3600)
    with pytest.raises(ValueError, match="retries"):
        reserve_phase(ended, "c" * 64, "drafts", ["a"], 3600)


def test_full_budget_is_shared_across_phases_and_an_overrun_blocks_more_spending():
    ledger = reserve_phase(empty_ledger("fixture"), "a" * 64, "drafts", ["a"], 40 * 3600)
    ended = finish_phase(ledger, "a" * 64, 40 * 3600, termination_confirmed=True)
    with pytest.raises(ValueError, match="aggregate"):
        reserve_phase(ended, "b" * 64, "judge", ["a"], 600)
    assert ended["conservative_compute_and_disk_estimate_usd"] < 30


def test_request_ceiling_and_approval_are_not_reset_per_phase():
    with pytest.raises(ValueError, match="approval"):
        empty_ledger("")
    ledger = reserve_phase(
        empty_ledger("fixture"), "a" * 64, "drafts", [str(i) for i in range(6000)], 600
    )
    ended = finish_phase(ledger, "a" * 64, 100, termination_confirmed=True)
    with pytest.raises(ValueError, match="request limit"):
        reserve_phase(ended, "b" * 64, "drafts", ["extra"], 600)
