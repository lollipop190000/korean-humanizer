"""Illustrative maximum-configured-work estimate, not a provider spending cap."""

from decimal import Decimal

from pydantic import Field

from korean_humanizer.models import Record


class APILine(Record):
    name: str
    requests: int = Field(ge=0)
    input_tokens_per_request: int = Field(ge=0)
    max_output_tokens_per_request: int = Field(ge=0)
    input_usd_per_million: Decimal = Field(ge=0)
    output_usd_per_million: Decimal = Field(ge=0)


class CostConfig(Record):
    pricing_status: str
    approved_for_execution: bool = False
    budget_usd: Decimal = Field(ge=0)
    api: list[APILine] = Field(default_factory=list)
    gpu_hourly_usd: Decimal = Field(default=Decimal(0), ge=0)
    gpu_max_hours: Decimal = Field(default=Decimal(0), ge=0)
    storage_gb: Decimal = Field(default=Decimal(0), ge=0)
    storage_days: Decimal = Field(default=Decimal(0), ge=0)
    storage_usd_per_gb_month: Decimal = Field(default=Decimal(0), ge=0)
    contingency_fraction: Decimal = Field(default=Decimal("0.2"), ge=0, le=1)


def estimate(config: CostConfig) -> dict:
    lines = []
    for line in config.api:
        amount = (
            Decimal(line.requests)
            * (
                line.input_tokens_per_request * line.input_usd_per_million
                + line.max_output_tokens_per_request * line.output_usd_per_million
            )
            / Decimal(1_000_000)
        )
        lines.append({"name": line.name, "usd": amount})
    lines.extend(
        [
            {"name": "gpu", "usd": config.gpu_hourly_usd * config.gpu_max_hours},
            {
                "name": "storage",
                "usd": config.storage_gb
                * config.storage_days
                * config.storage_usd_per_gb_month
                / Decimal(30),
            },
        ]
    )
    subtotal = sum((line["usd"] for line in lines), Decimal(0))
    total = subtotal * (1 + config.contingency_fraction)
    return {
        "pricing_status": config.pricing_status,
        "lines": [{"name": line["name"], "usd": str(line["usd"])} for line in lines],
        "subtotal_usd": str(subtotal),
        "total_with_contingency_usd": str(total),
        "budget_usd": str(config.budget_usd),
        "within_budget": total <= config.budget_usd,
        "execution_enabled": False,
        "note": (
            "Estimate only; provider execution, spending caps, tax and egress are not implemented."
        ),
    }
