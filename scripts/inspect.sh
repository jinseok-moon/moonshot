#!/usr/bin/env bash
# Static kernel inspection without ncu (unavailable on some boxes, ERR_NVGPUCTRPERM):
#   - ptxas -v : registers / shared memory / spills
#   - cuobjdump -sass : confirm the instructions that matter (HMMA, LDGSTS) actually emit
#
# Usage: scripts/inspect.sh <source.cu> [arch]
#   arch defaults to the current GPU's compute capability (via nvidia-smi), else sm_86.
set -euo pipefail

SRC="${1:?usage: scripts/inspect.sh <source.cu> [arch]}"

detect_arch() {
  local cap
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
  if [[ -n "$cap" ]]; then echo "sm_${cap}"; else echo "sm_86"; fi
}
ARCH="${2:-$(detect_arch)}"

if [[ ! -f "$SRC" ]]; then
  echo "no such file: $SRC" >&2
  exit 1
fi

INCLUDE="-Icsrc/include"

echo "=== ptxas -v ($ARCH) : $SRC ==="
nvcc -arch="$ARCH" $INCLUDE -Xptxas -v -cubin -o /tmp/moonshot_inspect.cubin "$SRC"

echo
echo "=== SASS grep (HMMA = tensor-core mma, LDGSTS = cp.async, IMMA = int8 mma) ==="
cuobjdump -sass /tmp/moonshot_inspect.cubin | grep -E 'HMMA|LDGSTS|IMMA' || \
  echo "  (no HMMA/LDGSTS/IMMA — expected for a scalar kernel or a pre-Ampere arch)"

rm -f /tmp/moonshot_inspect.cubin
