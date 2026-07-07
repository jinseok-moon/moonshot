#!/usr/bin/env bash
# Static kernel inspection without ncu (unavailable on the target box, ERR_NVGPUCTRPERM):
#   - ptxas -v : registers / shared memory / spills
#   - cuobjdump -sass : confirm the instructions that matter (HMMA, LDGSTS) actually emit
#
# Usage: scripts/inspect.sh <source.cu> [arch]
#   arch defaults to sm_86 (RTX 3090).
set -euo pipefail

SRC="${1:?usage: scripts/inspect.sh <source.cu> [arch]}"
ARCH="${2:-sm_86}"

if [[ ! -f "$SRC" ]]; then
  echo "no such file: $SRC" >&2
  exit 1
fi

INCLUDE="-Icsrc/include"

echo "=== ptxas -v ($ARCH) : $SRC ==="
nvcc -arch="$ARCH" $INCLUDE -Xptxas -v -cubin -o /tmp/moonshot_inspect.cubin "$SRC"

echo
echo "=== SASS grep (HMMA = tensor-core mma, LDGSTS = cp.async) ==="
cuobjdump -sass /tmp/moonshot_inspect.cubin | grep -E 'HMMA|LDGSTS|IMMA' || \
  echo "  (no HMMA/LDGSTS/IMMA — expected for a scalar kernel)"

rm -f /tmp/moonshot_inspect.cubin
