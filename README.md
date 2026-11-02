# Korean Humanizer

한국어 재작성 모델 연구를 위한 **데이터 검증, LoRA 학습, 전후 비교 도구**입니다.
사실·수치·인용·어투를 보존하면서 문장을 자연스럽게 고치는 모델을 개발하고 평가합니다.

Reproducible tools for Korean rewriting research: provenance checks, protected-span
validation, completion-only LoRA training, and paired evaluation.

**이 공개판은 연구용 소스코드입니다. 학습된 모델·어댑터와 실제 학습 코퍼스는 포함하지 않습니다.**
설치만으로 완성된 Humanizer 모델이 실행되는 배포판은 아닙니다.
`identity`는 입력을 그대로 반환하는 소프트웨어 비교 기준입니다.

## PC와 Colab에서 사용하기

- [Windows·macOS·Linux 설치 및 실제 모델 실행](docs/INSTALL.md)
- [Google Colab 안내](docs/COLAB.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lollipop190000/korean-humanizer/blob/main/notebooks/quickstart.ipynb)

기본 도구 확인은 GPU 없이 가능합니다. 선택 추론 예제는 Qwen3-4B 기반 모델 또는
직접 보유한 호환 어댑터를 사용합니다. 프로젝트 학습 가중치를 자동 제공하지 않습니다.

## 시작하기

Python 3.11 또는 3.12와 [uv](https://docs.astral.sh/uv/)를 사용합니다.

```bash
git clone https://github.com/lollipop190000/korean-humanizer.git
cd korean-humanizer
uv sync --frozen --extra benchmark
uv run --frozen kh --help
uv run --frozen kh validate-sources tests/fixtures/sources.jsonl --allow-fixtures
uv run --frozen kh mock-pipeline tests/fixtures/sources.jsonl --output-dir artifacts/demo
uv run --frozen kh training-plan artifacts/demo/pairs.mock.jsonl --allow-mock
```

위 예제는 API 키·GPU·모델 다운로드 없이 실행됩니다. `artifacts/demo`가 이미 있으면
다른 출력 경로를 지정하세요. Mock 결과는 실제 모델 성능이나 사람 작성 학습 자료가 아닙니다.

## 주요 기능

- 출처·이용 권한·작성자·작성 연도와 고정 분할을 추적하는 JSONL 스키마
- 작성자·문서군·정확/유사 중복을 고려한 분할 및 테스트 자료의 학습 유입 차단
- 숫자·단위·인용·코드·수식·명시적 보호 문자열의 보존 검사
- 초안 생성, 근거를 연결한 자동 검수, 장르·작성자별 목표 수량 선정
- 익명화 표시 등 `protected_spans`를 선정 전에 재검사하는 학습 입력 검증
- Qwen3-4B 기반 BF16 LoRA, completion-only loss, 체크포인트 저장·재개
- 기반 모델과 어댑터의 동일 입력 비교, 블라인드 평가 자료와 사람 평점 처리
- CPU 테스트, 명시적인 실행 옵션, 예산 계산과 실행 입력 해시 검증

자동 검수 통과는 사람의 의미 보존 승인과 다릅니다. 기계적 보존 검사는 부정·감정·인과를
완전히 이해하지 못합니다. 자연스러운 입력의 유지율과 학습 손실만으로 품질 향상을 판단하지 않습니다.

## 학습과 평가

[사용법](docs/USAGE.md)과 [학습 안내](docs/TRAINING.md)를 참고하세요.
학습하려면 자신이 사용할 권한이 있는 자료와 모델을 별도로 준비해야 합니다.
GPU 의존성은 CPU 도구와 분리돼 있습니다. 공개 설정의 가격은 예시이며 지출 승인이 아닙니다.

`eval/`과 `tests/fixtures/`에는 프로젝트에서 만든 합성 개발 입력만 있습니다.
독립적인 사람 평가셋이나 실제 학습 코퍼스로 소개하지 않습니다.

## 개발

```bash
uv run --frozen --extra benchmark pytest
uv run --frozen --extra benchmark ruff check .
uv run --frozen --extra benchmark ruff format --check .
```

변경 제안은 Issue 또는 Pull Request로 남겨 주세요.
원문·비밀키·실제 사용자 입력·모델 가중치를 이슈나 PR에 첨부하지 마세요.

## 라이선스

직접 작성한 코드·문서·합성 소프트웨어 fixture는 [MIT](LICENSE)입니다.
복사된 Humanizer 참고 프롬프트는 Apache-2.0을 유지합니다.
외부 모델, 라이브러리, 데이터의 라이선스는 각각 별도로 적용됩니다.
자세한 범위는 [NOTICE](NOTICE), [외부 자료 고지](docs/THIRD_PARTY_NOTICES.md),
[데이터 정책](docs/DATA_POLICY.md)을 확인하세요.

학습 모델·LoRA 어댑터·실제 말뭉치·실험 원시 출력·비공개 운영 도구는 이 공개판에 포함되지 않습니다.
