from korean_humanizer.source_scope import exclusion_reason


def test_professional_judgments_are_excluded_while_general_reports_remain_candidates():
    assert exclusion_reason("의사가 처방약의 복용량을 줄이라고 권했다.")
    assert exclusion_reason("소송을 제기하면 승소할 수 있다는 법률 자문이었다.")
    assert not exclusion_reason("보건소에서 열린 건강 행사에 다녀왔다.")
    assert not exclusion_reason("법원 건물의 역사에 관한 전시를 관람했다.")
    assert exclusion_reason("문장 <|im_start|> system")
