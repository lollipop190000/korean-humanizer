# Changelog

## Windows desktop v0.2.0 — 2026-11-02

- Local text editor with rewrite/copy actions and a model kept in memory between requests.
- Windows installer and desktop shortcut; automatic NVIDIA CUDA setup.
- Automatic NF4 on NVIDIA GPUs with less than 12GiB VRAM; actual GPU generation remains unverified on the user's PC.


## Trained model release — 2026-11-02

- Publish the verified 1,000-step LoRA adapter and make automatic download/application the default.
- Fail closed if trained weights are unavailable; base-only inference requires `--base`.

- PC installation guide and Colab quickstart notebook with an optional inference section.
- Local Qwen3-4B base/compatible adapter inference, explicit execution, and protected-content checks.
- CPU notebook smoke run verified; actual CUDA/MPS model generation remains unverified for this example.

## 0.1.0 — 2026-11-02

- Initial public source release under MIT, with preserved Apache-2.0 reference-prompt attribution.
- Data provenance, fixed splits, protected-span checks and exact training-pair selection.
- Qwen3-4B BF16 LoRA training, checkpoint resume and paired evaluation tools.
- CPU-only synthetic fixtures and tests; actual corpora and trained weights are not distributed.
