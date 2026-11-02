"""Explicit inference modes; these do not change or retrain model weights."""

MODES = ("standard", "strong")

STRONG_INSTRUCTION = (
    "다음 글을 같은 뜻의 자연스러운 한국어 글로 새로 쓰세요. 교정이 아니라 적극적인 재작성입니다. "
    "원문을 통째로 복사하거나 공백·어미·동의어 몇 개만 바꾸는 데 그치지 마세요. "
    "바꿀 수 있는 일반 서술은 다른 표현과 어순으로 구성하고, 필요하면 문장을 나누거나 합치세요. "
    "상투적인 도입과 반복되는 연결 표현은 의미를 잃지 않는 범위에서 자연스럽게 풀어 쓰세요. "
    "핵심 정보와 논리 흐름을 빠뜨리지 말고, 요약하거나 새로운 정보·의견·사례를 덧붙이지 마세요. "
    "누가 무엇을 했는지, 시간, 부정, 조건, 인과관계와 불확실성의 정도를 유지하세요. "
    "고유명사, 숫자, 단위, 인용, URL, 수식과 코드는 그대로 보존하세요. "
    "존댓말·반말, 격식과 화자의 관점은 유지하세요. "
    "보호할 표현만 있는 경우에는 억지로 바꾸지 마세요. "
    "설명, 제목, 평가, 따옴표 포장 없이 새로 쓴 본문만 출력하세요."
)


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError("재작성 모드는 기본 다듬기 또는 적극 재작성 중에서 선택해 주세요.")
    return mode


def instruction_for(standard_instruction, mode):
    validate_mode(mode)
    return STRONG_INSTRUCTION if mode == "strong" else standard_instruction


def generation_options(mode):
    validate_mode(mode)
    if mode == "strong":
        # Match the existing research comparison's sampling settings. Quality is unverified.
        return {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
    return {"do_sample": False}
