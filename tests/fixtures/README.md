# Synthetic fixtures

이 디렉터리의 한국어 문장은 프로젝트의 소프트웨어 테스트를 위해 AI가 새로 작성한 예시입니다.
실제 저작물에서 수집하지 않았고, 사람 원문·2019년 이전 자료·최종 평가 정답으로 주장하지 않습니다.
source_year는 null, origin은 synthetic_fixture입니다. 권한 메타데이터는 테스트 경로를 검증하기 위한 것입니다.
실제 외부 전송은 수행하지 않으며 third_party_transfer는 pending으로 둡니다.

sources.jsonl: 6개 장르, 24개 독립 문서. 고유명사·한자·숫자·단위·코드·수식·인용 포함.
evaluation_cases.jsonl: 루브릭을 설명하는 12개 설계 예시. 학습 또는 독립 holdout으로 쓰지 않습니다.
Mock는 실제 모델 성능을 평가하지 않습니다.
