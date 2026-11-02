"""Build SFT tokens on CPU; pending research needs a matching policy and explicit opt-in."""

import argparse
import json
from pathlib import Path

from korean_humanizer.sft import prepare_sft

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pairs", type=Path, required=True)
parser.add_argument("--metadata", type=Path, required=True)
parser.add_argument("--config", type=Path, default=Path("configs/sft-pilot.v1.json"))
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--allow-pending-research", action="store_true")
args = parser.parse_args()
print(
    json.dumps(
        prepare_sft(
            args.pairs,
            args.metadata,
            args.config,
            args.output_dir,
            allow_pending_research=args.allow_pending_research,
        ),
        ensure_ascii=False,
        indent=2,
    )
)
