# 학습과 체크포인트

학습된 어댑터를 사용하려면 [설치 안내](INSTALL.md)를 따르세요.
이 문서는 자신의 권한 있는 데이터로 새 학습을 수행하는 개발자를 위한 안내입니다.
학습 원문은 배포하지 않으므로 원래 학습 과정을 데이터까지 동일하게 재현할 수는 없습니다.

## 지원하는 학습 환경

현재 실행기는 Qwen3-4B 고정 revision과 BF16 LoRA를 사용합니다.
GPU 환경은 PyTorch 2.6.0+cu124, Transformers 4.51.3을 검사하며 BF16 지원 CUDA GPU가 필요합니다.
Python 3.11 환경을 권장합니다. GPU 패키지는 `requirements/sft-gpu.txt`에 분리돼 있습니다.
CPU 도구 설치는 GPU 환경 설치를 대신하지 않습니다. 설치 후 실제 전이 의존성도 기록하세요.

모델 revision과 토크나이저 해시는 `configs/baselines/qwen3-4b.json` 및 토크나이저 metadata로
고정합니다. 모델 이용 조건은 [Qwen3-4B 모델 저장소](https://huggingface.co/Qwen/Qwen3-4B)에서 확인하세요.

## 기본 준비

```bash
uv run --frozen --extra benchmark python scripts/fetch_tokenizers.py configs/baselines/qwen3-4b.json --output-dir artifacts/tokenizer-4b
uv run --frozen --extra benchmark python scripts/prepare_sft.py --pairs path/to/pairs.jsonl --metadata artifacts/tokenizer-4b --config configs/sft-pilot.v1.json --output-dir artifacts/sft-bundle
uv run --frozen --extra benchmark python scripts/train_sft.py --bundle artifacts/sft-bundle
```

첫 명령은 토크나이저를 다운로드합니다. 자료 검증을 통과해야 bundle이 만들어집니다.
마지막 명령은 기본 dry-run입니다. 실제 GPU 실행은 같은 bundle의 manifest 해시와
`--execute`, 출력 경로를 명시해야 합니다. 상세 옵션은 `train_sft.py --help`를 확인하세요.

`pending_research` 실험은 그 정책을 명시한 설정과 준비 단계의
`--allow-pending-research` 옵션을 모두 요구합니다. 공개 설정 자체는 실제 자료의 이용권이나
지출 승인이 아닙니다. 검증을 통과시키려고 실제 검수 상태를 변경하지 마세요.

## 학습 레시피

`sft-scale-001.v1.json`은 train 2,000/dev 200, 최대 1,000스텝, gradient accumulation 4,
학습률 5e-5, LoRA rank 16/alpha 32/dropout 0.05, 최대 길이 2,048토큰 예시입니다.
프롬프트와 padding을 loss에서 제외하고 정답 completion과 EOS에만 loss를 적용합니다.

100스텝마다 평가·저장합니다. `--stop-after-step`로 중간 저장 후
`--resume-checkpoint`로 같은 스케줄과 체크포인트에서 이어갈 수 있습니다.
체크포인트의 입력 manifest와 파일 해시가 맞아야 합니다.
완료 여부는 `result-step-*.json`의 `global_step`, `planned_steps`, `completed`로 확인합니다.

## 전후 비교

`scale_train_bundle.py`는 동일한 평가 입력을 동결합니다. `compare_sft.py`는 완성된
기반 모델/어댑터의 응답을 생성합니다. 입력·요청 해시를 확인하고 잘린 응답, 숫자·인용 보존,
자연 입력의 불필요한 변경, 지연시간을 함께 비교하세요.
손실 감소는 문장 품질 향상의 충분한 증거가 아닙니다. 실제 사람의 의미·자연스러움 평가는
별도로 기록해야 합니다. 공개 개발 입력은 최종 독립 평가셋을 대신하지 않습니다.
