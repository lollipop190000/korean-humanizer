# Third-party material

The instruction and separator in `configs/baselines/humanizer.json` reproduce
`jialinyyzz/humanizer/prompt_format.json` at revision
`2e57b6528a4070c33f3e09ca24e3325a84272f68`, published by the Humanizer project
(Stephen Yu / jialinyyzz / sgaofen). The model card states Apache License 2.0 for its code and weights.
The prompt is unchanged; the surrounding candidate, sampling and revision configuration is ours.

Source: https://huggingface.co/jialinyyzz/humanizer/blob/2e57b6528a4070c33f3e09ca24e3325a84272f68/prompt_format.json

The upstream LICENSE and NOTICE are preserved verbatim in
`licenses/humanizer-LICENSE.txt` and `licenses/humanizer-NOTICE.txt`.
The upstream notice describes its own Gemma-based models, not this project's Qwen training recipe. A standard Apache-2.0 copy is also in `licenses/Apache-2.0.txt`.
Original project code is MIT-licensed. The upstream prompt portions remain Apache-2.0 licensed. Neither license grants rights to third-party training corpora or makes excluded model weights available.


## Dependencies and model references

Dependencies are installed separately, not vendored or relicensed by this repository.
Pydantic and lm-format-enforcer use MIT; Jinja uses BSD-3-Clause; tokenizers uses Apache-2.0.
The GPU stack (Transformers, Accelerate, PEFT, safetensors and huggingface-hub) retains its
Apache-2.0 licenses; PyTorch retains its own BSD-style license and third-party notices.
Consult the pinned packages for authoritative license texts and bundled dependency notices.

Qwen/Qwen3-4B and Qwen/Qwen3-8B are referenced model dependencies, not distributed weights.
Their official model repositories provide their applicable licenses and model cards.
A reference to another model or a downloadable corpus does not put it under MIT.

The files in eval/ and tests/fixtures/ are original synthetic development material under MIT.
No NIKL corpus, private derived training pair, trained adapter, operational backup or raw
research response is included in this source release.
