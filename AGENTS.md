# Contributor instructions

- This is the public source repository. Keep private corpora, model weights, credentials and raw user outputs out of Git.
- MIT applies to original code; preserve NOTICE and upstream Apache-2.0 attribution.
- Fixtures and mock results are synthetic software tests, not model-quality evidence.
- Preserve protected spans, data splits and review provenance. Do not turn automatic passes into human acceptance.
- GPU/network operations require explicit execution options and the operator's own permissions and budget.
- Check with `uv run --frozen --extra benchmark pytest`, `ruff check .` and `ruff format --check .` using the same uv environment.
