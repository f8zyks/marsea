#!/usr/bin/env bash
# Acceptance tests T0-T16 (spec Sec. 11).  T0 needs the backbone in the HF cache and a GPU; the rest run on CPU.
set -e
cd "$(dirname "$0")/.."
MARSEA_DEBUG=1 .venv/bin/python -m pytest tests/ -q -s "$@"
