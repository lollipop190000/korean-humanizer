"""Download the trained project adapter and verify every pinned file hash."""

import argparse

from korean_humanizer.model_download import download_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", default="artifacts/models/nikl-practical-001")
args = parser.parse_args()
print(download_model(args.output_dir).resolve())
