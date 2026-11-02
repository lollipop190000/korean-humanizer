"""Conservative exclusions from the reviewed initial text scope, not semantic acceptance."""

import re

PROFESSIONAL_JUDGMENT = re.compile(
    r"복용량|투약|투여량|진단\s*기준|처방(?:전|약|을|받)|치료법|약물\s*상호작용"
    r"|법률\s*자문|소송.{0,20}(?:제기|승소|패소)|(?:형사|민사)(?:상|상의)?\s*(?:책임|처벌)"
    r"|조정\s*결정|신청인의.{0,40}주장을\s*기각"
)


def exclusion_reason(text):
    if PROFESSIONAL_JUDGMENT.search(text):
        return "professional_medical_or_legal_judgment_outside_initial_scope"
    if any(marker in text for marker in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]):
        return "reserved_model_control_marker"
    return ""


def scale_year_allowed(source):
    """Retain old publisher years; include the user's pre-2019 NIKL supplement."""
    if source.source_year in {2014, 2015, 2016, 2017, 2018}:
        return True
    return (
        source.source_doc_id.startswith(("nikl/", "nikl-np-v1.2/"))
        and source.source_year is not None
        and 1000 <= source.source_year < 2019
    )
