# 데이터 정책과 스키마

`Source`는 source_doc_id, author_group, source_group, genre, text, provenance, origin,
source_year, rights, protected_spans를 가집니다. 실제 작성 날짜의 근거는 provenance가 연결하는 비공개 카탈로그에 둡니다.
`source_year`를 적었다는 사실만으로 날짜를 검증한 것으로 간주하지 않습니다.
사용 허가 증빙은 사람이 확인하고, 프로그램은 필요한 상태/참조가 존재하는지 검증합니다.

## 권한

training, derivative_creation, third_party_transfer, redistribution을 독립적으로 approved/pending/denied로 기록합니다.
기본값은 pending입니다. 학습·파생 생성 허가와 evidence_ref가 없으면 사용을 거부합니다.
외부 API 전송은 별도 검사합니다. 다운로드 가능·공개 게시·데이터셋 전체 라이선스만으로 권한을 추정하지 않습니다.
합성 초안·심사 출력의 학습 이용 조건도 별도 검토합니다. 사용자 승인과 제3자 이용권은 서로 대체하지 않습니다.

## 날짜와 품질

초기 생산용 export는 확인된 2019년 이전 사람 원문을 대상으로 합니다.
최근 글도 연구상 배제 대상은 아닙니다. `validate-sources --allow-later-years`로 별도 후보를 조사할 수 있으며,
생산 export 정책 변경은 해당 자료·연도·품질 근거를 결정 기록에 남긴 뒤 구현합니다.
fixture는 source_year=null이며 실제 사람 글이라는 주장을 하지 않습니다.
문체·개인정보·복사·의미 변화 검수는 필수 후속 절차입니다. P0의 권한 필드는 법률 판단 자동화가 아닙니다.

## 분할

AI 초안 생성 전에 분할합니다. source_doc_id는 레코드별 고유 ID이고 같은 원문에서 잘라낸 조각은
동일 source_group을 사용합니다. author_group은 동일 작성자를 묶으며 source_group은 같은 문서/연재 등
누출 가능성이 있는 출처 단위입니다. 말뭉치 전체 이름을 모든 source_group에 넣으면 한 그룹으로 묶입니다.
작성자를 모르면 같은 사람일 가능성을 무시한 가짜 독립 ID를 만들지 말고 출처 단위 정책을 검토합니다.

현재 구현은 작성자/출처/정규화 동일 텍스트/유사 중복을 연결하고 전이적 연결요소를 분할합니다.
30자 이상 문장은 5자 shingle을 공유하는 후보 중 Jaccard>=0.85 또는 문자 배열 유사도>=0.95이면 묶습니다.
이는 짧은 한국어 문장의 한 음절 변경 때문에 shingle 일치율이 크게 떨어지는 경우를 보완합니다.
분할은 seed와 구성 레코드에 대해 결정적입니다. 데이터 추가 시 재분할하지 말고 기존 분할 manifest를 유지합니다.
작은 자료는 일부 split이 비거나 불균형할 수 있습니다. 그룹을 깨서 채우지 않고 자료를 보완합니다.
유사 중복은 같은 split에 묶습니다. 중복 제거 여부와 대표 레코드 선정은 사람이 검토하며 자동 삭제하지 않습니다.
후보 비교가 많으면 계산량이 커집니다. 2만~3만 쌍 확장 전에 성능 측정과 인덱스 개선이 필요합니다.

## 합성과 학습

2026-10-22 사용자는 설문 반영 전에 소규모 첫 학습을 진행하도록 지시했습니다.
별도 `pending_research` 설정과 명시적 준비 옵션에서 실제 pending Pair를 연구용으로 사용할 수 있습니다.
설문 점수나 원본 검수 상태를 바꾸지 않습니다. 생산 export의 accepted 요구와
권한·기계 보존·출처·비 Mock·누출·test 금지는 유지합니다. 범위: FIRST_RESEARCH_SFT.md.

Mock는 synthetic_fixture만 허용하며 test split은 생성하지 않습니다.
실제 pair는 생성 모델·프롬프트·seed·review_status·review_ref를 기록합니다.
생산 SFT export에는 human 원문과 accepted 상태 및 검수 참조가 필요합니다. Mock를 허용하면 파일 전체가 mock이어야 합니다.
export 시 서로 관련된 원문이 train/dev 사이에 중복되는지 다시 검사합니다.
보호 정보 검사를 통과해도 의미가 같다는 뜻은 아닙니다. 최종 accepted 판정은 별도 심사/사람 검수 근거가 필요합니다.

원문·파생 글·평가 정답표는 비공개 저장소와 Git에서 제외한 artifacts 등에 둡니다.
생산용 데이터셋에는 해시, 사용 조건, 보존/삭제 정책, 생성 비용, 검수자 기록을 연결합니다.
