import pytest

from korean_humanizer.protection import check_protection


@pytest.mark.parametrize(
    ("original", "output"),
    [
        ("온도는 25 ℃다.", "온도는 26 ℃다."),
        ("37명이 참여했다.", "37%가 참여했다."),
        ("CNN을 사용했다.", "RNN을 사용했다."),
        ("“내일 갈게”라고 말했다.", "“오늘 갈게”라고 말했다."),
        ("식은 $x + y = 3$이다.", "식은 $x - y = 3$이다."),
        ("코드는 `return x`다.", "코드는 `return y`다."),
        ("결과는 같다.", "결과는 2배다."),
        ("12명 중 12명이 응답했다.", "12명이 응답했다."),
        ("“할 수 있다”라고 했다.", "“할수있다”라고 했다."),
    ],
)
def test_changes_are_visible(original, output):
    assert not check_protection(original, output)["mechanical_pass"]


def test_korean_particles_and_unit_spacing_allowed():
    result = check_protection("CNN이 25 ℃에서 작동한다.", "25℃에서 CNN을 사용한다.")
    assert result["mechanical_pass"]
    assert result["semantic_status"] == "not_assessed"


def test_mechanical_check_does_not_claim_to_detect_negation():
    result = check_protection("효과가 있을 수 있다.", "효과가 없다.")
    assert result["semantic_status"] == "not_assessed"
    # Explicit protection can be supplied, but it is not a general semantic checker.
    assert not check_protection("효과가 있을 수 있다.", "효과가 없다.", ["있을 수 있다"])[
        "mechanical_pass"
    ]


def test_invalid_span_and_empty_result_fail():
    assert not check_protection("문장", "문장", ["없는 이름"])["mechanical_pass"]
    assert not check_protection("문장", "")["mechanical_pass"]
