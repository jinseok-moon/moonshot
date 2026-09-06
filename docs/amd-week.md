---
title: CUDA to AMD — one-week kernel study
status: planned
audience: [human, agent]
updated: 2026-09-06
tags: [amd, rocm, hip, curriculum, gemm]
related: [roadmap.md, hardware-and-measurement.md]
---

# CUDA → AMD: 1주일 집중 실습

CUDA kernel optimization 경험을 AMD architecture와 ROCm으로 옮긴다.
CUDA 기초는 반복하지 않고, wave execution, LDS, VGPR/SGPR, MFMA에서 실제로
달라지는 부분을 비교한다. 성능의 원인을 설명하는 것이 목표다.

## 현재 단계: GPU와 ROCm 환경 선정

아직 서버를 선택하지 않았다. Day 1 시작 전에 MI-series 후보의 architecture,
임대 비용, ROCm 이미지, hardware-counter 접근 가능 여부를 확인한다.
선택 시점의 AMD 공식 지원 문서와 공급자 재고를 확인하고 GPU를 결정한다.
이전 별도 저장소의 GPU 추천 순위는 검증된 선정 결과로 취급하지 않는다.

서버의 GPU model, `gfx*`, ROCm/HIP/compiler 버전, GPU 노출 상태, profiler와
counter 권한을 확인한 후 Day 1로 넘어간다. 실제 명령·API·metric을 제시할 때는
설치 버전에 맞는 AMD 공식 문서를 확인한다. GPU 실습은 ROCm 서버에서 수행한다.

## 진행 방식

사용자가 “Day 1 시작하자”처럼 요청하면 해당 날짜의 작은 실험 하나부터 진행한다.

개념 설명 → 작은 실습 → 사용자 실행 → 결과 확인 → profiler/ISA 분석 → 다음 단계.

분석은 가설과 관측 사실을 구분한다. 최적화 완료 조건은 correctness,
reference 대비 측정값, 사람이 설명할 수 있는 architecture reasoning이다.
전체 강의나 최적화 ladder를 미리 완성하지 않는다.

| Day | 집중할 내용 | 실험/산출물 |
| --- | --- | --- |
| 1 | HIP runtime, stream/event, launch, error handling | vector add와 reduction; timing은 후속 실습에서 추가 |
| 2 | CDNA CU, wavefront, workgroup, LDS, VGPR/SGPR, scalar/vector execution, scheduling, cache | reduction workgroup 크기를 바꾸며 resource usage와 성능 비교 |
| 3 | ROCm profiling | coalesced/strided, register pressure, LDS 사용량을 통제한 비교 |
| 4 | BF16 GEMM, global→LDS→register, vector access, MFMA | naive→tiled→MFMA를 별도 파일로 보존; generated ISA 확인 |
| 5 | CK / CK Tile | descriptor, tile distribution, wave mapping, intrawave/interwave scheduling; CK GEMM 비교 |
| 6 | hipBLASLt | heuristic, selected solution, workspace, online/offline tuning; small-M/large-M 비교 |
| 7 | GEMM mini project | naive/tiled/MFMA/hipBLASLt를 같은 조건에서 검증·측정·분석 |

Day 6–7 shape는 `M ∈ {1,16,128,512,2048,4096}`, `N=K=4096`을 출발점으로 한다.
dtype, layout, transpose, alpha/beta, correctness tolerance를 통일한다.
latency, TFLOPS, reference 대비 성능과 함께 VGPR/SGPR/LDS, occupancy,
bandwidth, cache, VALU/MFMA 지표를 수집 가능한 범위에서 기록한다.
수집하지 못한 지표는 0으로 쓰지 않고 미측정으로 표시한다.

## 코드와 결과 위치

- 독립 HIP 실습: [`csrc/studies/amd/`](../csrc/studies/amd/README.md).
- 실험 노트: `docs/kernels/`에 frontmatter와 reference 대비 결과를 남긴다.
- 로컬 빌드·원시 profiler 출력: Git에서 제외되는 `build/amd-study/` 아래에 둔다.
- engine custom op 연결: backend capability gate를 갖춘 뒤 기존 extension seam으로 통합한다.

이 집중 과정은 [roadmap](roadmap.md)과 연결된 독립 HIP 학습 과정이다.
기존 장기 engine 계획은 유지하며, AMD custom op 통합은 후속 작업으로 진행한다.
