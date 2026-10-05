#!/bin/sh
# Run from the repository root after uv sync --frozen.
set -eu
output=${1:-runs/additional-seeds}
nice -n 19 timeout 2h uv run --frozen yoked-plasticity --config configs/additional-seeds.json --output "$output"
nice -n 19 uv run --frozen python tools/verify_artifacts.py "$output"
nice -n 19 uv run --frozen python tools/summarize_work.py "$output"
