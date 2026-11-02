import pytest

from korean_humanizer.models import Rights, Source


@pytest.fixture
def source():
    return Source(
        source_doc_id="doc-1",
        author_group="author-1",
        source_group="document-1",
        genre="report",
        text="실험에는 CNN을 사용했다. 온도는 25 ℃였으며 효과가 있을 수 있다.",
        provenance="fixture:generated-for-tests",
        origin="synthetic_fixture",
        rights=Rights(
            training="approved",
            derivative_creation="approved",
            evidence_ref="tests/fixtures/README.md",
        ),
        protected_spans=["CNN"],
    )
