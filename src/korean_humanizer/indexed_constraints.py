"""Schema-constrained evidence output; verdicts remain the model's decision."""

from korean_humanizer.automated_quality import IndexedJudgment
from korean_humanizer.revision_protocols import evidence_units


def judgment_schema(source, draft):
    schema = IndexedJudgment.model_json_schema()
    for field, text, side in [
        ("source_evidence_ids", source, "S"),
        ("draft_evidence_ids", draft, "D"),
        ("genre_evidence_ids", source, "S"),
    ]:
        ids = [
            u["id"] for u in evidence_units(text, side) if sum(c.isalnum() for c in u["text"]) >= 2
        ]
        if not ids:
            raise ValueError("no substantive evidence units")
        schema["properties"][field]["items"] = {"type": "string", "enum": ids}
    return schema


def batch_prefix_filter(tokenizer_data, schemas):
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer

    enforcers = [TokenEnforcer(tokenizer_data, JsonSchemaParser(s)) for s in schemas]

    def allowed(batch_id, input_ids):
        return enforcers[batch_id].get_allowed_tokens(input_ids.tolist()).allowed_tokens

    return allowed
