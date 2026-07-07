# Template op

Copy-me example for adding a CUDA custom op to the engine. Demonstrates the full seam with a
trivial `out = x * alpha + beta`.

## Add a real op

1. `cp -r csrc/ops/template csrc/ops/<name>`; rename the file and the op.
2. Replace the kernel + launcher. Keep the three-part shape:
   - `__global__` kernel,
   - a launcher returning `torch::Tensor` (validate dtype/device, `CUDA_CHECK_LAST()`),
   - `TORCH_LIBRARY_FRAGMENT` (schema) + `TORCH_LIBRARY_IMPL(..., CUDA, ...)` (bind).
3. The op is now `torch.ops.moonshot.<name>`. Add/confirm its dispatcher in
   [`moonshot/ops.py`](../../../moonshot/ops.py) — prefer the custom op, keep the torch fallback.
4. Build on the 3090 (`pip install -e .`), bench against the fallback, and write
   `docs/kernels/<name>.md` with the number **and** the why-fast note.

## Ladder discipline

An optimization step is a **new** `kernel_N` file, never an in-place edit of a prior rung.
The ladder of rungs is the story an interviewer follows.

## Notes

- Registration uses `TORCH_LIBRARY_FRAGMENT` so multiple ops can share the `moonshot`
  library namespace across files.
- Add a `Meta`/fake impl later (`register_fake`) so the op survives `torch.compile` (Phase 6).
