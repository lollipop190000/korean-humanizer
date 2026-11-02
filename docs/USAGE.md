# 사용법

## CPU 도구 확인

```bash
uv sync --frozen --extra benchmark
uv run --frozen kh --help
uv run --frozen kh benchmark-validate eval/dev_seed.v1.jsonl
uv run --frozen kh benchmark-prepare eval/dev_seed.v1.jsonl --spec configs/baselines/qwen3-4b.json --output-dir artifacts/requests --allow-unreviewed
```

`--allow-unreviewed`는 합성 개발 입력을 준비하기 위한 명시적 옵션입니다. 사람 검수 완료를
뜻하지 않습니다. `kh mock-pipeline`도 모델 호출 없이 소프트웨어 연결을 검사합니다.

## 자료 준비 흐름

1. `Source`/`AssignedSource` 스키마에 출처와 권한 근거를 기록합니다.
2. 작성자·문서군을 기준으로 train/dev/test를 고정합니다.
3. `prepare_scale_drafts.py`로 고정 생성 입력을 준비합니다.
4. `generate_scale_batch.py`와 `prepare_scale_judge.py`로 생성·검수 단계를 분리합니다.
5. `screen_scale_outputs.py`로 출력·요청·근거 연결을 검사합니다.
6. `finalize_scale_pairs.py`로 수량·장르·작성자 및 명시적 보호 문자열을 검증합니다.
7. `prepare_sft.py` 또는 `scale_train_bundle.py`로 학습 입력을 동결합니다.

각 스크립트의 실제 옵션은 `uv run --extra benchmark python scripts/<이름>.py --help`로
확인할 수 있습니다. 이 설명은 데이터 이용권이나 유료 실행을 승인하지 않습니다.
실행 권한을 나타내는 JSON 필드만으로 권한을 새로 만들 수는 없습니다.

## 입력 형식

`Source`는 원문, 출처, 작성자 그룹, 작성 연도, 권한 근거와 `protected_spans`를 갖습니다.
`Pair`는 원문, 생성된 입력, 고정 분할, 생성 모델 정보와 검수 상태를 연결합니다.
스키마는 `src/korean_humanizer/models.py`, 합성 예제는 `tests/fixtures/`에 있습니다.

`pending`은 자동 검수나 연구용 선정 상태이며 `accepted`와 다릅니다.
생산용 학습 내보내기에는 사람 검수와 근거가 필요합니다. 별도 `pending_research` 설정은
미검수 연구임을 유지하며, 사람 검수나 생산 준비 완료로 바꾸지 않습니다.

## 결과 해석

`mechanical_pass`는 수치·인용·보호 문자열 등의 기계적 검사입니다.
사실·의미·어투와 자연스러움은 별도로 평가해야 합니다. 미완료·실패·길이 종료도 함께 보고합니다.
같은 입력을 기반 모델과 어댑터에 제공하고, 사람 평가 시 모델 이름을 가립니다.
