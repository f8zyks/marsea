#!/usr/bin/env bash
# Handover bundle: HEAD, working-tree status, and a sha256 manifest of EVERY file the tarball ships, plus a tarball
# of marsea/ scripts/ tests/ alongside it.  Run after the round's commit, never before.
# The manifest used to list *.py and *.sh only, so two PDFs under scripts/ shipped unlisted; the manifest is the
# provenance instrument and covers the whole tree (review e6ca44f H).
set -e
cd "$(dirname "$0")/.."
{ git rev-parse HEAD; git status --porcelain; echo '---'; \
  find marsea scripts tests -type f -not -path '*/__pycache__/*' -not -name '*.pyc' -not -path '*/runs/*' \
    | sort | xargs sha256sum; } > HANDOFF.txt
tar czf ~/marsea_$(git rev-parse --short HEAD).tar.gz \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='runs' \
    marsea scripts tests HANDOFF.txt
echo "=== ~/marsea_$(git rev-parse --short HEAD).tar.gz"; ls -lh ~/marsea_$(git rev-parse --short HEAD).tar.gz
