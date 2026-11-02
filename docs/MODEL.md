---
base_model: Qwen/Qwen3-4B
library_name: peft
license: mit
language:
- ko
pipeline_tag: text-generation
---
# Korean Humanizer — nikl-practical-001

1,000스텝 학습을 완료한 한국어 재작성 연구용 LoRA 어댑터입니다. 원래 Qwen3-4B에
이 어댑터를 적용해야 이번 프로젝트의 학습 결과를 사용합니다. 모델 전체가 아닌 약 126MiB의
추가 가중치이며, 기반 모델은 별도로 다운로드합니다.

학습된 추가 가중치는 MIT로 공개합니다. 기반 모델과 토크나이저의 Apache-2.0 고지를 함께 유지합니다.

## 학습과 검증

- 기반 모델: Qwen/Qwen3-4B, 고정 revision 1cfa9a7208912126459214e8b04321603b3df60c.
- 실제 학습: BF16 LoRA, rank 16/alpha 32/dropout 0.05, 1,000스텝, epoch 2.
- 자료: train 2,000/dev 200. 기계적 검증·자동 검수 통과 연구용 자료이며 사람 승인 완료가 아닙니다.
- 국립국어원 비출판물·신문 말뭉치와 이전에 확보한 출처 추적 자료를 활용했습니다.
  자료 이용 사실을 명시하며 원문과 학습 예제는 배포하지 않습니다.
- 172입력/344응답 기반 모델·어댑터 비교를 완료했습니다. 합성 개발 입력과 자연 입력을
  사용한 연구 비교이며 독립적인 사람 품질 평가 완료를 뜻하지 않습니다.
- 사람 의미·어투·자연스러움 검수는 아직 남아 있습니다. 성능 향상이나 모든 사실의 보존을
  보장하지 않습니다. 숫자·부정·조건·인용을 직접 확인하세요.

## 사용

공개 저장소의 설치 안내를 따르고 어댑터를 내려받은 뒤 로컬 실행기에 적용합니다.
https://github.com/lollipop190000/korean-humanizer
토크나이저는 이 패키지의 고정 파일을 사용하며 thinking은 끕니다.
원래 연구의 샘플링 비교와 간단 실행 예제의 greedy 출력은 다를 수 있습니다.
긴 글보다 문단 단위로 사용하고 출력 길이 종료를 확인하세요.

## 라이선스와 포함 범위

코드 및 직접 학습한 추가 가중치는 MIT로 공개합니다.
기반 Qwen3-4B 및 토크나이저에는 Apache-2.0이 별도로 적용됩니다.
LICENSE, NOTICE, BASE_MODEL_LICENSE.txt를 함께 보존하세요.
훈련 원문·생성 원문·optimizer·trainer 상태·training_args.bin·비밀값은 포함하지 않습니다.
