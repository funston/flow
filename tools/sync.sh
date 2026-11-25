#!/bin/bash
# Copy the flow workflow out of a BACH checkout into this repository.
# The code is developed there; this repository is the part that can be
# installed on its own.
#
#   tools/sync.sh ~/work/bach
set -euo pipefail
SRC="${1:?usage: tools/sync.sh <path to bach checkout>}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
FILES=(
  bin/bach-flow
  src/bach/__init__.py
  src/bach/ops/__init__.py
  src/bach/ops/workflow.py
  src/bach/ops/linear.py
  src/bach/ops/flow_cli.py
  integrations/claude-code/install.py
  test/unit/test_ops_workflow.py
  test/unit/test_ops_linear.py
  test/unit/test_flow_cli.py
  test/unit/test_flow_pr_review.py
  LICENSE
)
cd "$SRC"
rsync -a --relative "${FILES[@]}" "$HERE/"
rsync -a --delete integrations/claude-code/helpers/ "$HERE/integrations/claude-code/helpers/"
echo "synced ${#FILES[@]} files and the helpers from $SRC"
