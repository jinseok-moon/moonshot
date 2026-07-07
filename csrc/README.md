# csrc/ — CUDA C++ custom ops

Kernels plug into the engine here, one op per directory, registered as
`torch.ops.moonshot.<name>` so [`moonshot/ops.py`](../moonshot/ops.py) can dispatch to them.

```
csrc/
  include/moonshot/     shared headers
    check.cuh           CUDA_CHECK error macro
    bench.cuh           CUDA-event median Timer (counter-free timing)
    verify.hpp          max_rel_error against a reference
  ops/
    template/           copy-me op: TORCH_LIBRARY registration + trivial kernel
    <name>/             one directory per real op (rmsnorm, gemm_bf16, attention, ...)
```

## Add a kernel

1. `cp -r csrc/ops/template csrc/ops/<name>` and rename.
2. Write the kernel; register with `TORCH_LIBRARY(moonshot, m)` (schema in `m.def`, CUDA
   impl in `m.impl(..., c10::DispatchKey::CUDA, ...)`).
3. Make sure `moonshot/ops.py` has a `<name>` dispatcher (custom-op-preferred, torch fallback).
4. Build on the 3090 and bench against the fallback.

## Build

Compiled via torch `cpp_extension` (`pip install -e .` from the repo root) on the target box
(`sm_86`, arch `80;86;89`). Until a kernel is added the package is pure Python.

**CUDA does not build on macOS** — edit and reason statically off-box; compile and run on
the 3090. `ncu` is unavailable (`ERR_NVGPUCTRPERM`); measure with CUDA events, `ptxas -v`,
`cuobjdump -sass`, `nsys`. See [`docs/hardware-and-measurement.md`](../docs/hardware-and-measurement.md).
