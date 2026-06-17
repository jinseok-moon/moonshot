**하드웨어:** RTX 3090 (Ampere, sm_86, 24GB) — bf16/int8 Tensor Core, `cp.async` 가능 / fp8·Hopper `wgmma`·TMA 불가
**툴체인:** CUDA 13.0, CMake, **Nsight Systems(`nsys`)** + CUDA events + `ptxas -v`/SASS
> ⚠️ **`ncu`(Nsight Compute)는 이 Vast 컨테이너에서 사용 불가** — `ERR_NVGPUCTRPERM`. GPU 하드웨어 perf counter 접근이 호스트 모듈 파라미터(`NVreg_RestrictProfilingToAdminUsers`)로 막혀 있고, unprivileged 컨테이너 내부에서 풀 수 없음. **`nsys`는 정상 작동**(tracing/CUPTI). 대체 프로파일링 방법은 Phase 1 참고.

> 원칙: **문제 → 벤치마크 → 프로파일 → 글/PR**. 매 단계가 (1) repo에 남는 코드, (2) 측정 수치, (3) 블로그/PR 형태 산출물을 동시에 만든다. 단순 공부 금지.

---

## 현재 위치 (이미 완료)
- fp32 GEMM 풀 최적화: naive → shared tiling → register tiling → **warptiling + float4** (`src/cuda/gemm/fp32/kernel_0..5`)
- bf16 Tensor Core GEMM (WMMA API, `bf16/kernel_1`)
- bank conflict / pinned memory / thread indexing 마이크로 토픽

→ 입문 단계는 끝. GEMM 단일 문제에 깊이 도달했으나 (a) 프로파일링 규율, (b) 추론 특화 커널(attention/quant), (c) Triton/프레임워크 통합이 비어 있음. 아래 7단계로 그 공백을 메운다.

---

## Phase 1 — 프로파일링 규율 + bf16 텐서코어 GEMM 사다리 ★ 시작점 (5~6주)
**여기가 출발선.** fp32 사다리는 끝냈으니, 같은 사다리를 **텐서코어용으로 다시** 오른다.
현재 bf16 트랙은 `kernel_0`(naive cast), `kernel_1`(WMMA·블록당 1 warp·shared 없음)뿐 → k2~k7까지 쌓는다.
연산이 텐서코어라 병목이 연산이 아니라 **메모리 레이턴시**로 옮겨가는 게 fp32와의 핵심 차이.
**`ncu`를 못 쓰므로** 하드웨어 카운터 없이 정량화하는 방법론을 같이 익힌다 (Simon Boehm 블로그도 ncu 없이 GFLOPs/% 기반).

**Week 1 — 벤치 하니스 구축 (`bench/`) + 측정 규율**
- [ ] CUDA events 타이머: warmup N회 → 측정 M회 중앙값, `cudaDeviceSynchronize` 정확히 배치
- [ ] GFLOPs(`2*M*N*K/t`) + achieved bandwidth(bytes/t) 유틸, **cuBLAS(bf16) baseline** 연결 → 모든 커널 **cuBLAS 대비 %** CSV 표
- [ ] 정확도 검증: bf16라 cuBLAS 결과 기준 `max rel err < 2e-2` (`src/common/verify.hpp` 활용)
- [ ] shape sweep: M=N=K ∈ {1024, 2048, 4096}, skinny(4096×4096×1024) 1개
- [ ] 측정 보조 루틴 상시화: `nvcc --ptxas-options=-v`(reg/smem/spill), `cuobjdump -sass`(inner loop), `cudaOccupancyMaxActiveBlocksPerMultiprocessor`(이론 occupancy), `nsys`(타임라인)
- [ ] 기존 `kernel_0`/`kernel_1`을 이 하니스로 측정 → 베이스라인 % 확보

**Week 2 — k2: shared memory tiling (WMMA)**
- [ ] A/B 타일을 shared로, **블록당 여러 warp**, 각 warp가 16×16 출력 fragment 담당
- [ ] global 로드 재사용 효과를 bandwidth 수치로 확인 (vs k1)

**Week 2~3 — k3: warp tiling**
- [ ] warp 하나가 **여러 WMMA fragment**(예: 2×2 / 4×4 타일) 계산 → register 재사용 + ILP↑
- [ ] fp32 `kernel_5` warptiling 구조의 텐서코어판. 타일 파라미터 템플릿화

**Week 3~4 — k4: cp.async 파이프라이닝 ★ (성능의 핵심)**
- [ ] `__pipeline_memcpy_async`(또는 `cp.async` PTX)로 global→shared 비동기 로드
- [ ] **double buffering**: 다음 K-타일 prefetch와 현재 타일 mma 오버랩 (smem 2배)
- [ ] 텐서코어 GEMM은 메모리 레이턴시 바운드 → 여기서 성능 대부분이 나옴을 nsys로 확인

**Week 4~5 — k5: bank-conflict-free smem + 벡터 로드**
- [ ] shared 레이아웃 padding/**swizzle**로 fragment 로드 시 bank conflict 제거 (기존 bank_conflict 실습 직결)
- [ ] 128-bit(=bf16 8개) 벡터 로드로 global→shared 대역폭 끌어올리기

**Week 5~6 — k6: `ldmatrix` + `mma.sync` PTX (WMMA 탈피)**
- [ ] WMMA 버리고 **`ldmatrix`로 smem→register 정확한 레이아웃 로드 + `mma.sync.aligned.m16n8k16`** 직접 호출
- [ ] CUTLASS/cuBLAS가 실제로 하는 방식. fragment 레이아웃을 손에 쥠 → Phase 3에서 그대로 재사용
- [ ] (스트레치) k7: 3+ 스테이지 cp.async 파이프라인 + 타일 크기 sweep

- [ ] **완료 기준: k6(또는 k7) ≥ 80% cuBLAS(bf16)** @ 4096³. 미달 시 병목을 SASS/occupancy/표로 설명할 수 있을 것
- **산출물:** `bench/` 하니스 + roofline plot + 블로그 1편 "Climbing the tensor-core GEMM ladder on Ampere (without ncu)"
- **참고:** Simon Boehm SGEMM 글, NVIDIA PTX ISA(`ldmatrix`/`mma`), CUTLASS efficient-gemm 문서, `cuda-samples`

---

## Phase 2 — 추론 커널 빌딩블록 프리미티브 (4주)
추론 커널은 결국 reduction/softmax/normalization의 조합. warp-level primitive를 손에 익힌다. 전부 **메모리 바운드**라 목표는 bandwidth 한계 근접.

**Week 1 — reduction 계층**
- [ ] `__shfl_down_sync` 기반 **warp reduce** (sum/max)
- [ ] shared mem 기반 **block reduce** (warp reduce 조합), grid-stride loop
- [ ] vs `thrust::reduce` 정확도·속도, achieved bandwidth가 peak(3090 ~936 GB/s)의 몇 %인지

**Week 2 — softmax**
- [ ] naive softmax (max → exp → sum → div, 3-pass)
- [ ] **online/2-pass softmax** (running max+sum 한 번에) — FlashAttention 빌드업
- [ ] row 길이별(128~32768) 벤치, vs `torch.softmax`

**Week 3 — normalization**
- [ ] **LayerNorm** (mean/var 동시 계산, Welford 또는 2-pass), backward는 Phase 5로 미룸
- [ ] **RMSNorm** (LLM 표준) + residual add fused 버전
- [ ] vectorized(float4/half2) 로드로 bandwidth 끌어올리기

**Week 4 — 정리**
- [ ] (옵션) inclusive/exclusive **scan** (Blelloch) 1개
- [ ] 모든 프리미티브 한 README 표에 bandwidth % 정리
- **산출물:** `src/cuda/primitives/` 라이브러리 + "메모리 바운드 커널 튜닝" 노트
- **참고:** NVIDIA "Faster Parallel Reductions" 슬라이드, `cub::BlockReduce` 소스

---

## Phase 3 — FlashAttention 직접 구현 ★ 키스톤 (6~8주)
**포트폴리오의 핵심.** 추론 직무 면접에서 가장 자주 검증. Phase 2의 online softmax가 여기서 합쳐진다.

**Week 1~2 — 수식 & 레퍼런스**
- [ ] online softmax + attention 재귀식 손으로 유도(노트), naive attention CUDA로 먼저 구현(정답지)
- [ ] PyTorch SDPA로 골든 텐서 덤프해 검증 파이프라인 구축

**Week 3~5 — fused forward 커널 (`src/cuda/attention/`)**
- [ ] Q·K·V 타일을 shared로, **fused QK^T → online softmax → ·V** 단일 커널 (bf16 입력, fp32 accumulate)
- [ ] block당 query 타일 처리, K/V를 시퀀스 따라 순회하며 running (m, l, acc) 갱신
- [ ] `cp.async` 더블버퍼(Phase 1 재사용), `mma.sync`/WMMA로 두 matmul 텐서코어화
- [ ] **causal mask**, head_dim ∈ {64, 128} 지원
- [ ] 완료 기준: 정확도 `< 2e-2`(bf16), 속도 **SDPA(flash backend) 대비 ≥ 60%**

**Week 6~8 — 추론 특화 & 마감**
- [ ] **decode 경로**(seq_len_q=1) + **KV-cache** + **GQA/MQA**(KV head 공유) — 실제 inference 핵심
- [ ] seq len {512, 2048, 8192} sweep, prefill vs decode 분리 벤치
- [ ] (스트레치) FlashAttention-2 대비 비교, 또는 sliding-window
- **산출물:** `src/cuda/attention/` + 상세 블로그(수식 유도 + 벤치 그래프). 이력서: "FlashAttention from scratch, X% of FA2 on Ampere"
- **참고:** FlashAttention 1/2 논문, `flash-attention` repo, `flashinfer` decode 커널

---

## Phase 4 — 양자화 GEMM 커널 (4주)
추론 최적화 채용 수요 폭발 영역. 3090 **int8 Tensor Core(`mma.sync.s8`)** + weight-only 4bit.

**Week 1 — int8 정수 GEMM**
- [ ] DP4A(`__dp4a`) 기반 int8 GEMM → `mma.sync.aligned.s8` 텐서코어 버전
- [ ] per-tensor/per-channel scale로 int32 accum → fp 역양자화, vs `cublasLtMatmul` int8

**Week 2~3 — W4A16 weight-only dequant GEMM (LLM 추론 표준)**
- [ ] 4bit weight 패킹 포맷 설계(2개/byte) + group-wise scale/zero-point
- [ ] 커널 내 **언팩 → bf16 dequant → bf16 GEMM** fuse (Marlin/llama.cpp 구조 참고)
- [ ] group size {64, 128} 정확도(perplexity 대신 max err)·속도 측정
- [ ] memory traffic 절감을 bandwidth 수치로 증명 (weight 1/4)

**Week 4 — 정리**
- [ ] activation 양자화까지 W4A8 한 변형(스트레치)
- **산출물:** `src/cuda/quant/` + "Ampere에서 4bit weight-only GEMM" 노트
- **참고:** Marlin 커널, llama.cpp `ggml-cuda` mmq, AutoGPTQ/AWQ 커널

---

## Phase 5 — Triton + PyTorch 통합 (프레임워크 백엔드 진입) (6주)
CUDA로 한 걸 Triton으로 다시 → 컴파일러 관점 + 생산성 + 프레임워크 통합. **DL 백엔드 직무의 핵심 증거물.**

**Week 1~2 — Triton 기초 & 재구현**
- [ ] Triton 튜토리얼(vector add → fused softmax → matmul) 완주
- [ ] Phase 2 프리미티브(softmax/RMSNorm) Triton 재구현, `@triton.autotune`으로 config 탐색
- [ ] 같은 문제 CUDA vs Triton 성능·생산성 비교 표

**Week 3~4 — Triton FlashAttention & matmul**
- [ ] Triton matmul + (스트레치) Triton FlashAttention forward
- [ ] autotune된 Triton 커널이 손튜닝 CUDA 대비 몇 %인지

**Week 5~6 — PyTorch 통합**
- [ ] **커스텀 op 등록**: `torch.library.custom_op` + `register_fake`(meta) → `torch.compile` 안에서 동작
- [ ] **autograd**: `torch.autograd.Function`으로 backward 연결 (RMSNorm 또는 attention)
- [ ] 네이티브 CUDA 커널을 `torch.utils.cpp_extension.load`로 바인딩 (Phase 3 커널 1개)
- [ ] `torch.compile` graph break 없이 통과하는지 확인
- **산출물:** pip 설치 가능 미니 패키지(예: `myattn` — 커스텀 fused op + 테스트 + 벤치)
- **참고:** Triton 공식 튜토리얼, PyTorch "Custom C++/CUDA Operators" 가이드

---

## Phase 6 — 오픈소스 기여 스프린트 (Phase 3부터 병행 시작)
실력의 사회적 증명 + 채용 레퍼런스. **목표: 9개월 내 merged PR 3개+.**

**상시 루틴 (Phase 3 시작 시점부터)**
- [ ] 타깃 1개 repo 골라 build from source + 테스트 통과시키기 (기여의 80%는 환경 셋업)
- [ ] `good first issue`·`help wanted`·perf 라벨 이슈 5개 정독, 1개 클레임
- [ ] 코드 읽기 → 작은 PR(문서/버그/벤치) 먼저로 리뷰어와 신뢰 쌓기

**진입 난이도 순 타깃**
1. **llama.cpp / ggml** — `ggml-cuda` 커널, perf/quant PR 활발, Ampere 친화 (현실적 첫 PR)
2. **tinygrad** — 공개 bounty(보상금) 존재, 학습 겸 보상
3. **FlashInfer** — attention 커널, Phase 3와 직결
4. **vLLM / SGLang** — `csrc` 커널(paged attention, quant), 난이도↑ 이력서 가치↑
5. **PyTorch ATen CUDA** — 최고 난도/권위

**마일스톤**
- [ ] PR #1 (≤6개월): 1번에서 작은 perf/버그 fix merged
- [ ] PR #2 (≤9개월): 1~3번에서 커널 최적화/추가 PR
- [ ] PR #3 (≤12개월): 3~4번에서 의미 있는 커널 기여

---

## Phase 7 — 마무리 & 취업 패키징 (마지막 4주)
- [ ] 포트폴리오 README/사이트: GEMM·FlashAttention·Quant·Triton 4축 + 벤치 그래프 + "재현 방법"
- [ ] 블로그 3~4편(Phase 1/3/4) 공개 (learning in public)
- [ ] 이력서 bullet: "X% of cuBLAS", "X% of FA2", "merged N PRs to llama.cpp/vLLM"
- [ ] 모의 면접 대비: "이 커널 왜 빠른가"를 화이트보드로 설명하는 연습(occupancy/roofline/타일링)
- [ ] 국내 타깃: **FuriosaAI · Rebellions · Moreh · SqueezeBits · Nota · Naver Cloud · Samsung SAIT** / 해외 추론팀

---

## 분기별 마일스톤
| 분기        | 내용                               | 산출물                                           |
| ----------- | ---------------------------------- | ------------------------------------------------ |
| Q1 (1~3M)   | Phase 1(bf16 GEMM 사다리 k2~k7) ~2 | 텐서코어 GEMM 벤치 블로그, 프리미티브 라이브러리 |
| Q2 (4~6M)   | Phase 3 + OSS PR 시작              | FlashAttention 구현 + 블로그, 첫 merged PR       |
| Q3 (7~9M)   | Phase 4~5                          | Quant 커널, Triton/PyTorch op 패키지, PR 2~3개   |
| Q4 (10~12M) | Phase 6~7                          | 추가 PR, 포트폴리오 패키징, 지원                 |

## 하드웨어 / 환경 메모
- 3090로 Phase 1~6 전부 가능 (Ampere int8/bf16 TC, cp.async)
- **`ncu` 사용 불가** (Vast unprivileged 컨테이너, `ERR_NVGPUCTRPERM`). `nsys`·CUDA events·`ptxas -v`·SASS로 대체. 하드웨어 카운터가 꼭 필요하면 profiling 허용된 박스에서 cross-check.
- **fp8 / Hopper wgmma / TMA**가 필요한 실험(최신 FA3, fp8 GEMM)은 Vast에서 **H100 주말 대여**로 별도 진행 (이때 ncu 가능 여부도 호스트 설정에 따라 다름 — 대여 시 확인)
- 멀티 GPU/통신(NCCL, tensor parallel)은 여력 되면 보너스 — 추론 분산에 가점

## 운영 팁
- 매 단계 끝에 `git tag` + 짧은 회고 커밋
- 벤치는 항상 cuBLAS/cuDNN/torch 레퍼런스 대비 %로 보고 (절대수치 X)
- "왜 빠른가"를 **GFLOPs/bandwidth 수치 + SASS/occupancy 근거 + nsys 타임라인**으로 설명할 수 있을 때만 다음 단계로 (ncu 스크린샷이 아니라 이 스택으로)
