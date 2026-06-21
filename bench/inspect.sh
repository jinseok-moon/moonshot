#!/usr/bin/env bash
# Static kernel inspection for the GEMM ladder (no ncu needed).
#   ptxas -v        : registers / shared mem / spills per kernel
#   cuobjdump -sass : inner-loop SASS (HMMA tensor-core ops)
# Usage: bench/inspect.sh [src-fragment] [target]
#   src-fragment : path fragment to match in compile_commands (default bf16/gemm.cu)
#   target       : CMake target / object dir name           (default gemm_bf16)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build"
FRAG="${1:-bf16/gemm.cu}"
TARGET="${2:-gemm_bf16}"

CC="${BUILD}/compile_commands.json"
[[ -f "$CC" ]] || { echo "configure first: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release"; exit 1; }

# Pull the nvcc command for the matching .cu out of compile_commands.json.
mapfile -t LINES < <(python3 - "$CC" "$FRAG" <<'PY'
import json, sys
cc, frag = sys.argv[1], sys.argv[2]
for e in json.load(open(cc)):
    if frag in e["file"]:
        print(e["file"]); print(e["directory"]); print(e["command"]); break
PY
)
SRC="${LINES[0]:-}"; DIR="${LINES[1]:-}"; CMD="${LINES[2]:-}"
[[ -n "$SRC" ]] || { echo "no .cu compile entry matching '$FRAG'"; exit 1; }

echo "### ptxas -v  ($(basename "$SRC"))"
# Re-run the exact compile command (from its build dir, where the .rsp files
# live) with verbose ptxas; discard the object.
( cd "$DIR" && eval "${CMD} -Xptxas=-v -o /tmp/inspect.o" ) 2>&1 \
  | grep -E "Compiling entry|registers|smem|spill|stack frame" || true

OBJ="${BUILD}/src/cuda/gemm/CMakeFiles/${TARGET}.dir/${FRAG}.o"
if [[ -f "$OBJ" ]]; then
  echo; echo "### tensor-core SASS (HMMA / shared-mem loads, first 12 lines)"
  cuobjdump -sass "$OBJ" | grep -iE "HMMA|LDS|LDGSTS" | head -12 || true
fi
