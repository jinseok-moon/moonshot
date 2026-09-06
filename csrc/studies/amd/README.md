# AMD HIP study

[1주일 학습 흐름](../../../docs/amd-week.md)의 독립 실행 예제.
현재 Day 1 vector add와 workgroup reduction 초안만 있다.
stream/event timing, profiling, GEMM은 이후 작은 실험으로 추가한다.
macOS에서는 ROCm 컴파일·GPU 실행을 검증하지 못했다.

빌드 진입점은 이 디렉터리의 `CMakeLists.txt`이며 Python package 빌드와 별개다.
GPU/ROCm 환경을 확인한 뒤 저장소 루트에서 실행한다.
`<gfx-target>`은 선택한 GPU의 실제 target으로 바꾼다.

```bash
cmake -S csrc/studies/amd -B build/amd-study -DCMAKE_HIP_ARCHITECTURES=<gfx-target>
cmake --build build/amd-study -j
./build/amd-study/day01_vector_add
./build/amd-study/day01_reduction
```

reduction은 block별 합을 GPU에서 계산하고 최종 합산을 CPU에서 수행한다.
현재 검증 입력은 모두 1인 FP32 배열이다. 성능 benchmark나 일반 입력 검증은
아직 포함하지 않는다. 최적화 시 이전 rung을 보존하고 새 파일을 추가한다.
