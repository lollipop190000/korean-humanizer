# PC 설치와 문장 다듬기

명령어 대신 글 입력 화면을 쓰려면 [Windows 설치 파일](DESKTOP.md)을 이용하세요.

[Colab에서 설치 없이 시작하기](COLAB.md) · [학습 안내](TRAINING.md)

이번 학습 결과는 **Qwen3-4B + nikl-practical-001 LoRA 어댑터**입니다.
설치는 실행 도구와 기반 모델을 준비하고, 추가 어댑터를 적용하는 과정입니다.
학습된 어댑터 약 126MiB와 고정 토크나이저를 합친 배포 파일은 약 141MiB입니다.
이 문서는 터미널 도구 설치 안내입니다. 별도의 [로컬 화면](DESKTOP.md)도 제공합니다.

[모델 릴리스](https://github.com/lollipop190000/korean-humanizer/releases/tag/model-nikl-practical-001)에 실제 학습 파일을 공개했습니다.
기본 실행은 어댑터를 자동으로 받고 검증합니다. 다운로드 실패 시 오류로 멈추며 기반 모델로
대체하지 않습니다. 직접 받은 어댑터는 `--adapter`로 사용 가능합니다.

## 1. 기본 도구 설치 (GPU 불필요)

Git과 [uv](https://docs.astral.sh/uv/getting-started/installation/)를 설치하세요.
Windows는 PowerShell, macOS/Linux는 터미널에서 아래 명령을 한 줄씩 실행합니다.
Python은 uv가 3.11로 준비하므로 시스템 Python을 바꿀 필요가 없습니다.

```bash
git clone https://github.com/lollipop190000/korean-humanizer.git
cd korean-humanizer
uv python install 3.11
uv sync --frozen --python 3.11 --extra benchmark
uv run --frozen kh --help
uv run --frozen kh validate-sources tests/fixtures/sources.jsonl --allow-fixtures
uv run --frozen kh mock-pipeline tests/fixtures/sources.jsonl --output-dir artifacts/demo
```

마지막 예제는 합성 데이터와 mock으로 도구 연결만 확인합니다. 모델 추론은 하지 않습니다.
출력 폴더가 이미 있으면 `artifacts/demo-2`처럼 새 이름을 쓰세요.
Windows에서 `uv`가 인식되지 않으면 설치 후 PowerShell을 다시 여세요.

## 2. 실제 학습 모델 실행

아래는 학습이 아닌 **단일 문장/짧은 글 추론**입니다. 처음에는 모델 다운로드가 필요합니다.
모델 파일용 여유 디스크 15GB 이상을 준비하세요. 메모리는 입력 길이·장치·라이브러리에 따라
달라지며, 대략 NVIDIA VRAM 12~16GB, Apple Silicon 통합 메모리 24GB 이상,
CPU RAM 32GB 이상을 권장합니다. CPU는 느립니다. 모든 장치에서 실측한 보장 사양은 아닙니다.
양자화는 이 예제에 포함하지 않습니다.

CPU 도구와 별도 가상 환경에 추론 의존성을 설치합니다. 프로젝트 루트에서 실행하세요.

### Windows (PowerShell)

```powershell
uv venv --python 3.11 .venv-inference
uv pip install --python .venv-inference/Scripts/python.exe -e . -r requirements/inference.txt
.venv-inference/Scripts/python.exe scripts/rewrite_local.py --help
```

### macOS / Linux

```bash
uv venv --python 3.11 .venv-inference
uv pip install --python .venv-inference/bin/python -e . -r requirements/inference.txt
.venv-inference/bin/python scripts/rewrite_local.py --help
```

편집기에서 `artifacts/input.txt`를 만들고 다듬을 글을 **UTF-8**로 저장합니다.
`artifacts` 폴더가 없으면 먼저 만드세요. 아래 예제는 macOS/Linux 기준입니다.

```bash
# 설정 확인: 모델을 내려받거나 실행하지 않습니다.
.venv-inference/bin/python scripts/rewrite_local.py --text-file artifacts/input.txt
# 실제 추론: 검증된 프로젝트 어댑터와 고정 버전 Qwen3-4B를 다운로드합니다.
.venv-inference/bin/python scripts/rewrite_local.py --text-file artifacts/input.txt --execute
```

Windows의 실제 실행 예제:

```powershell
.venv-inference/Scripts/python.exe scripts/rewrite_local.py --text-file artifacts/input.txt --execute
```

결과는 JSON의 `output`에 표시됩니다. 기본 모델은 `candidate: nikl-practical-001`입니다.
`--execute`에서 어댑터·설정·토크나이저를 내려받고 모든 파일의 크기와 SHA-256을 검증합니다.
이후에는 검증한 파일을 재사용합니다. 첫 실행에만 어댑터와 기반 모델 다운로드가 필요합니다.

학습 전 기반 모델과 비교하려면 같은 명령에 **`--base`**를 명시하세요. 이때만
`candidate: qwen3-4b-base`가 표시됩니다. `--base`와 `--adapter`는 함께 사용할 수 없습니다.
`finish_reason: length`이면 출력 길이 한도로 잘렸으므로 `--max-new-tokens 512`를 추가하거나
입력을 줄이세요. 기본 출력은 256토큰, 입력은 프롬프트 포함 2,048토큰으로 제한합니다.
숫자·인용 검사는 함께 출력하지만 의미·어투·자연스러움은 직접 확인해야 합니다.
탐욕적 생성 예제이므로 고정 연구 비교의 샘플링 설정과 결과가 같지는 않습니다.

장치는 자동 선택합니다(CUDA → Apple MPS → CPU). `--device cpu`, `--device cuda`,
`--device mps`로 지정할 수 있습니다. CUDA가 감지되지 않으면 NVIDIA 드라이버와
[PyTorch 설치 안내](https://pytorch.org/get-started/locally/)를 확인하세요.
이 예제는 PyTorch 2.6.0을 사용합니다. 호환 NVIDIA 드라이버가 있는 Windows/Linux에서는
추론 환경의 Python 경로를 지정해 `uv pip install --python <경로> torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`
로 CUDA 12.4 빌드를 설치할 수 있습니다. 다른 프로젝트의 Python 환경에는 설치하지 마세요.
메모리 부족이면 입력/출력 길이를 줄이거나 더 큰 메모리 장치를 사용하세요.

## 3. 직접 보유한 LoRA 어댑터 사용

Qwen/Qwen3-4B의 동일 기반 버전으로 학습한 **로컬 폴더**만 지정합니다.
`adapter_config.json`과 `adapter_model.safetensors`가 필요합니다.
기본 실행은 프로젝트 어댑터를 자동 다운로드합니다. `--adapter`는 이미 받은 파일이나
자신이 사용할 권한이 있는 다른 호환 어댑터로 바꾸고 싶을 때만 필요합니다.

```bash
.venv-inference/bin/python scripts/rewrite_local.py --text-file artifacts/input.txt --adapter artifacts/my-adapter --execute
```

`candidate: local-adapter`로 표시합니다. 기반 모델명과 기록된 revision을 검사하며,
revision이 기록되지 않은 외부 어댑터의 정확한 호환성은 사용자가 확인해야 합니다.
어댑터 사용권과 기반 모델 조건은 소스코드의 MIT 라이선스와 별도입니다.
입력·출력은 로컬에서 처리하고 추론 API에 보내지 않습니다. 첫 모델 다운로드에는 인터넷이 필요합니다.

## 다운로드 오류

모델 릴리스가 미공개이거나 인터넷 접근이 막혀 있으면 실행을 중단합니다. 같은 명령에
`--base`를 추가하는 것은 다른 모델을 쓰는 것이며 다운로드 오류의 해결책이 아닙니다.
해시 불일치 파일은 자동 덮어쓰지 않습니다. 해당 파일을 다른 위치로 옮긴 뒤 다시 실행하세요.

## 검증 범위

CPU 설치·예제·dry-run·노트북 기본 셀은 자동 검사합니다. CUDA/MPS의 실제 모델 생성과
Colab GPU 배정은 이 설치 안내 추가 작업에서 실행하지 않았습니다. GPU 패키지 환경과
하드웨어가 맞아야 하며, 이 예제를 기존 학습 모델의 성능 검증으로 소개하지 않습니다.
