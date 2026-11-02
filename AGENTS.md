# Contributor instructions

- This is the public source repository. Keep private corpora, credentials and raw user outputs out of Git. Model weights belong in a verified release asset, never source history.
- MIT applies to original code; preserve NOTICE and upstream Apache-2.0 attribution.
- Fixtures and mock results are synthetic software tests, not model-quality evidence.
- Preserve protected spans, data splits and review provenance. Do not turn automatic passes into human acceptance.
- GPU/network operations require explicit execution options and the operator's own permissions and budget.
- Check with `uv run --frozen --extra benchmark pytest`, `ruff check .` and `ruff format --check .` using the same uv environment.

- The user requested publication of the trained adapter. On 2026-11-02 the user confirmed that provider approval includes public download and redistribution of trained weights. Publish the verified adapter with MIT terms and preserve base-model/tokenizer Apache-2.0 notices. Do not request the same confirmation again.
