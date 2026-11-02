from korean_humanizer.scale_selection import select_exact


def test_quotas_reassign_flexible_author_instead_of_greedy_failure():
    rows = [
        dict(case_id="a1", author_group="a", genre="experience"),
        dict(case_id="a2", author_group="a", genre="information"),
        dict(case_id="b1", author_group="b", genre="experience"),
    ]
    assert set(select_exact(rows, {"experience": 1, "information": 1}, author_cap=1)) == {
        "a2",
        "b1",
    }


def test_no_padding_when_genre_or_author_supply_is_insufficient():
    rows = [
        dict(case_id="a1", author_group="a", genre="experience"),
        dict(case_id="a2", author_group="a", genre="information"),
    ]
    assert select_exact(rows, {"experience": 1, "information": 1}, author_cap=1) is None
    assert select_exact(rows, {"experience": 1, "opinion": 1}, author_cap=20) is None
