# Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling

## 0. 한눈에 보기

- **출처**: ISCA 2020 / Mahmoud Khairy, Zhesheng Shen, Tor M. Aamodt, Timothy G. Rogers / Purdue University & UBC
- **링크**: [IEEE DOI](https://doi.org/10.1109/ISCA45697.2020.00047) | [저자 PDF](https://mkhairy.github.io/Docs/Accel-Sim.pdf)
- **분류 태그**: #GPU-simulation #trace-driven #mISA #GPGPU #accelerator #validation #HW-SW-codesign
- **TL;DR**: NVIDIA SASS 트레이스 기반 frontend + 대폭 강화된 memory system 모델로 GPGPU-Sim 3.x 대비 cycle error를 94%→15%로 줄인 오픈소스 GPU 시뮬레이션 프레임워크.
- **핵심 기여**:
  1. NVBit 기반 SASS 트레이서 → ISA-independent intermediate representation으로 변환하는 **trace-driven frontend** (cuDNN 등 closed-source 라이브러리 시뮬레이션 최초 지원)
  2. GPGPU-Sim 4.0: sub-warp sectored coalescer, adaptive streaming L1, IPOLY L2 hashing, HBM dual-bus, write-validate sub-sector write 정책 등 **memory system 대폭 상세화**
  3. 38개 microbenchmark + 자동 파라미터 tuner + correlation 시각화 도구로 이루어진 **validation 인프라**
  4. Kepler→Turing 4세대 GPU를 단일 프레임워크로 모델링하는 **multi-generation 유연성** 실증
  5. 구식 베이스라인이 숨기는 두 가지 연구 기회(L1 throughput 병목 소멸, FR-FCFS 효과 극대화)를 보여주는 **case study**
- **핵심 통찰**: 시뮬레이터가 현대 하드웨어를 충분히 정밀하게 모델링하지 못하면, 연구가 발견하는 "병목"과 "기회"가 실리콘에 존재하지 않는 허상일 수 있다.

---

## 1. 문제 정의 & 동기

### 왜 지금인가

GPU 마이크로아키텍처는 제품 세대마다 undocumented 방식으로 빠르게 진화한다. Volta부터 도입된 Tensor Core, HBM, sub-core 모델, SASS ISA 변경 등은 기존 오픈소스 시뮬레이터들이 추적하지 못하는 수준의 변화다. 결과적으로 학계 연구자들은 다음 세 가지 문제에 직면한다:

1. **기능적 문제**: 매년 새 mISA가 등장하는데 functional model을 유지하는 비용이 너무 크다. cuDNN 같은 hand-tuned SASS 바이너리는 PTX 표현 자체가 없어 기존 simulator에서 아예 실행 불가.
2. **정확도 문제**: Volta QV100에서 GPGPU-Sim 3.x는 cycle error 94%, correlation 0.87이다(논문). 이 수준의 오차는 "어떤 아이디어가 효과적인가"를 잘못 판단하게 만든다.
3. **검증 방법론 문제**: 새 하드웨어 모델을 추가할 때 체계적인 validation 절차가 없어 수동 작업이 방대하다.

### 기존 접근의 Gap

| 시뮬레이터 | ISA 지원 | Volta 정확도 | 폐쇄 라이브러리 |
|---|---|---|---|
| GPGPU-Sim 3.x | PTX vISA + GT200 mISA | cycle MAE 94% | ✗ |
| Gem5-APU | AMD vISA+mISA | N/A | ✗ |
| MGPUSim | AMD vISA+mISA | MAE 5.5% (AMD only) | ✗ |
| MacSim | Intel trace-driven | N/A | ✗ |
| Multi2Sim | AMD/NVIDIA 구형 mISA | MAE 19% | ✗ |
| **Accel-Sim** | **NVBit-generated SASS** | **cycle MAE 15%** | **✓** |

(논문 Table I 기반)

---

## 2. 핵심 아이디어

NVIDIA는 binary compatibility를 위해 vISA(PTX)와 mISA(SASS)를 분리한다. 기존 시뮬레이터들은 PTX를 emulate하여 실행하는데, 
이는 (1) 실제 register allocation, instruction scheduling이 반영되지 않고, 
(2) 매년 새 mISA의 functional model을 직접 구현해야 한다는 이중 부담을 낳는다. 

Accel-Sim의 핵심 통찰은 

"functional correctness는 포기하고 timing만 replay한다"는 것이다. 

실제 GPU 위에서 NVBit으로 mISA 트레이스를 수집하면, 
새 ISA가 나와도 functional model 없이 timing simulation이 가능하다. 

대신 accuracy는 성능 모델(GPGPU-Sim 4.0)의 상세도가 결정하므로, 
저자들은 memory system을 중심으로 현대 GPU 아키텍처를 역공학하여 반영한다.

---

## 3. 배경 (필요한 것만, 압축)

### vISA vs mISA의 구조적 차이

- **PTX (vISA)**: 가상 무한 레지스터, 추상화된 메모리 모델, nvcc가 생성하고 드라이버가 mISA로 JIT 컴파일
- **SASS (mISA)**: 실제 하드웨어 register 할당, 실제 instruction scheduling, 하드웨어마다 다른 opcode. sm_70(Volta), sm_75(Turing) 등 compute capability별로 다름

### Sub-core 모델 (Volta 이후)

Volta 이전은 SM 내 warp scheduler들이 register file과 execution unit을 공유. Volta부터 **sub-core** 개념 도입: warp scheduler + 전용 RF + 전용 EU의 묶음이 isolated 단위로 동작, instruction cache와 memory subsystem만 SM 전체가 공유. (논문 인용, Raihan et al. ISPASS'19에서 처음 모델링)

### Sectored Cache

128B cache line을 32B sector 4개로 분할. tag는 128B 단위이되, 실제 data fetch는 32B 단위. 이로써 uncoalesced access에서의 over-fetch를 줄이고 CAM(content-address memory) 태그 전력 감소.

### IPOLY (Irreducible Polynomial) Hashing

$2^n$ 개의 L2 bank를 가진 HBM에서 stride $2^k$ 패턴의 접근은 특정 bank로 집중된다(partition camping). Galois field 기반의 irreducible polynomial interleaving으로 $2^n$ stride에 대해 conflict-free를 보장.

---

## 4. 제안 방법 상세

### 4.1 Flexible Frontend — 트레이스 생성부터 ISA-independent 표현까지

```
[실제 GPU 실행]
    CUDA binary (cuDNN 포함)
        ↓  NVBit LD_PRELOAD
    SASS 명령별 콜백 → PC, active_mask, reg_info, 메모리 주소 수집
        ↓  base+stride 압축
    .trace 파일 (per-kernel, per-warp)
        ↓  Accel-Sim frontend
    ISA-independent intermediate representation
        ↓
    GPGPU-Sim 4.0 performance model
```

**ISA-independent intermediate representation의 필드** (논문 Table II):

| 필드 | 내용 | trace 모드 | exec 모드 |
|---|---|---|---|
| PC | 명령 주소 | 트레이스에서 읽음 | emulation 계산 |
| active\_mask | warp 내 active thread 비트맵 | 트레이스에서 읽음 | emulation 계산 |
| Reg info (dst, src) | 접근 레지스터 목록 | ISA def 파싱 | PTX 파싱 |
| Exec unit | 실행할 EU 종류 | ISA def 파일 매핑 | PTX 타입 매핑 |
| Memory addresses | 실제 접근 주소 배열 | 트레이스에서 읽음 | emulation 계산 |
| Memory scope | global/shared/local + cache 정책 | 트레이스에서 읽음 | PTX 한정자 |

**ISA Def 파일**: 새 SASS opcode → execution unit 매핑. 공개 문서([NVIDIA CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html))에서 추론 가능. 새 세대 GPU 지원 시 이 파일과 HW Def 파일만 추가하면 됨.

**trace 파일 구조**: per-kernel 디렉토리 + `kernelslist` 파일 (kernel 실행 순서, launch 파라미터). 내부는 per-warp 명령열 + 메모리 접근 주소.

**base+stride 압축**: 메모리 접근 패턴이 규칙적인 경우 32개 주소 전체 대신 base + stride로 저장. 비규칙적 패턴은 전체 주소 저장. 압축률 약 20× (예: 2.5TB raw → 125GB compressed, CUTLASS 기준).

**simulation-gating**: cycle-driven 방식에서 "할 일 없는 컴포넌트"는 tick을 skip. 각 컴포넌트(core, EU, cache, DRAM channel)마다 status state 유지. Trace-driven 모드에서 GPGPU-Sim 3.x 대비 4.3× 속도 향상, 12.5 KIPS (kilo warp instructions/sec) 달성.

**kernel-based checkpointing**: 긴 initialization kernel을 skip하고 hot 커널만 시뮬레이션. trace-driven 모드에서만 가능.

### 4.2 GPGPU-Sim 4.0 Performance Model 상세화

**Sub-warp, Sectored Memory Coalescer**

Kepler는 32-thread warp 전체를 128B 단위로 coalesce. Pascal/Volta/Turing은 8-thread sub-warp 단위로 32B sector에 접근:

- warp 내 thread 0-7: 1st sub-warp → 최대 8개의 32B sector 요청 생성
- warp 내 thread 8-15: 2nd sub-warp → 독립 처리
- ...

이전 모델(32-thread coalescer)은 L1 read accesses를 4× 과소계산 (128B line = 4 × 32B sector인데 sector 단위로 count하는 하드웨어와 불일치). Accel-Sim은 post-coalescer access에서 NRMSE=0.00, Correl=1.00 달성.

**Adaptive, Streaming L1 Cache**

- 용량: Volta 기준 shared memory와 L1D를 합산 최대 128KB. kernel이 shared memory를 사용하지 않으면 전부 L1D에 할당.
- Streaming cache: MSHR 엔트리 수 대폭 증가. 할당된 L1D 용량과 무관하게 MSHR throughput 동일.
- Banked (4 banks): 병렬 접근 지원
- Latency: 28 cycles (Volta), GPGPU-Sim 3.x의 가정(전혀 다른 설정)과 큰 차이

**Write-Validate Sub-sector Write Policy** (논문에서 발견한 undocumented feature)

기존 GPGPU-Sim: naive write policy → write마다 128B line 전체를 DRAM에서 fetch (per-cacheline fetch on write). 이로 인해 DRAM reads 대폭 과다 추정.

실제 L2 동작 (Accel-Sim이 역공학으로 발견):

```
write miss → sector에 byte 기록 + byte-level write mask 설정
                  ↓
sector가 modified 상태로 전환 (DRAM fetch 없음)
                  ↓
이후 해당 sector에 read 요청 도착
  → write_mask 완전? (모든 byte 기록됨)
      YES → 그대로 read, DRAM 접근 없음
      NO  → DRAM에서 sector fetch + modified bytes merge
```

L2 write: 최소 1B 단위 (sub-32B 가능)
L2 read: 최소 32B sector 단위
→ partial write 후 read 시에만 DRAM fetch 발생. DRAM bandwidth 절약.

**IPOLY L2 Partition Hashing**

HBM 8-channel 구성에서 address bit[3:0] (캐시라인 내 offset)을 제외한 상위 비트 중, row bits에서 무작위 선택된 비트들을 XOR하여 L2 bank index 결정.

$$\text{bank\_idx} = \text{addr}[k:3] \oplus P(\text{addr}[\text{row\_bits}])$$

$P$는 Galois field의 irreducible polynomial. $2^n$ stride 접근에 대해 conflict-free 보장. (논문 인용: Rau ISCA'91)

**HBM / GDDR6 Memory Model**

- HBM: dual-bus interface (row command + column command 동시 발행 → bank parallelism 향상), 상세 timing, per-bank refresh
- GDDR6: 기존 GDDR5 대비 read/write buffer 분리, advanced xor-based bank indexing 추가
- CPU-GPU memory copy engine 모델: 모든 DRAM 접근은 L2를 통과하므로 이를 통한 copy traffic도 모델링

**Tensor Core 모델링**

Volta: WMMA (warp-level matrix multiply accumulate) PTX → GPGPU-Sim에서 추상적으로 지원
Accel-Sim: SASS-level HMMA instruction 직접 시뮬레이션 (Raihan et al. ISPASS'19 분석 기반). microbenchmark로 latency/throughput 측정 후 configuration 파일에 지정.

### 4.3 Tuner & Microbenchmark Suite

38개 microbenchmark 카테고리:

| 카테고리 | 측정 대상 |
|---|---|
| Latency | L1, L2, shared mem, DRAM (data type별: float/double/128b vector) |
| Bandwidth | L1, L2, shared mem, DRAM (포화 측정) |
| Cache geometry | associativity, bank 수, sector 크기, write policy |
| Coalescing | access granularity, sub-warp 크기 |
| Execution unit | SPU/DPU/SFU/Tensor Core latency/throughput |
| Streaming | MSHR 수, streaming cache 동작 |

**Pointer-chasing 패턴으로 L1 latency 측정**:

```c
// HW_def.h에서 L1_ARRAY_SIZE 정의 (공개 문서 기반)
for (i = 0; i < L1_ARRAY_SIZE-1; i++)
    data[i] = (uint64_t)(data + i + 1);  // 포인터 체인 구성

// 타이밍 루프: ld.global.ca (L1 cached)로 ITERS=32768회 pointer-chase
asm volatile ("ld.global.ca.u64 ptr1, [ptr0];");
```

→ 컴파일된 SASS: `LDG.E.64 R12, [R10]` 연속 의존성 체인, latency = (stopClk - startClk) / ITERS

자동 tuner가 결정하지 못하는 4가지 파라미터(warp scheduling, memory scheduling, L2 interleaving granularity, L2 hash function)는 bandwidth microbenchmark 조합 시뮬레이션으로 최고 상관계수를 주는 조합 선택.

### 4.4 Correlator

- nvprof / nsight-cli로 수집한 hardware counter를 시뮬레이터 counter와 1:1 비교
- 자동 correlation plot 생성 (scatter plot, x=HW, y=Sim)
- 수치: Correl (Karl Pearson 변동계수 = σ/μ), MAE (cycle/instruction/IPC), NRMSE (나머지 counter)

---

## 5. 시스템 · HW 측면

### 5.1 HW vs SW 분담

| 구성요소 | 역할 | 비고 |
|---|---|---|
| 실제 GPU (NVBit) | 트레이스 생성 (functional ground truth) | 1회 실행, 결과를 파일로 저장 |
| ISA Def 파일 | opcode → EU 매핑 | 사용자가 공개 문서로 작성 |
| HW Def 파일 | SM 수, warp 크기 등 기초 파라미터 | 공개 사양서에서 추출 |
| Accel-Sim Tuner | microbenchmark 결과 → config 자동 생성 | SW |
| GPGPU-Sim 4.0 | cycle-accurate timing simulation | SW, 싱글스레드 |
| Correlator | HW counter ↔ sim counter 비교 | SW, 사람이 해석 |

### 5.2 Simulation 환경

- **Simulation 방식**: cycle-driven (매 cycle 모든 컴포넌트 tick), simulation-gating으로 idle 컴포넌트 skip
- **RTL/합성**: 없음. 순수 SW performance model
- **Simulator 속도**: trace-driven 12.5 KIPS, exec-driven 6 KIPS (논문 Table I)
- **멀티스레딩**: 미지원. 싱글스레드 (논문 Table I)
- **참고**: 후속 연구(외부, arXiv 2502.14691)에서 병렬화 시도 — 하지만 단순 병렬화 시 determinism 문제 존재

### 5.3 트레이스 물리적 특성

| 벤치마크 | 비압축 크기 | 압축 후 |
|---|---|---|
| Rodinia 3.1 (13개) | 302 GB | 15 GB |
| CUDA SDK (10개) | 18 GB | 1.2 GB |
| Parboil (8개) | 250 GB | 12 GB |
| Polybench (10개) | 743 GB | 33 GB |
| CUTLASS (20개 입력) | 2.5 TB | 125 GB |
| Deepbench (60개) | 2.6 TB | 130 GB |
| **합계** | **~6.2 TB** | **~317 GB** |

트레이스 생성 시간: 1개 GPU로 약 48시간 (논문). 각 GPU generation별로 별도 생성 필요 (sm_70 traces를 Turing sm_75에 재사용 — NVBit Turing 지원이 2020년 여름 예정이었기 때문).

---

## 6. 🎯⭐ 실험 셋업 & 재현 관점

### 6.1 검증 대상 GPU

| GPU | Architecture | sm | DRAM | #SM |
|---|---|---|---|---|
| NVIDIA GeForce GTX TITAN | Kepler | sm_35 | GDDR5 | 14 |
| NVIDIA TITAN X | Pascal | sm_62 | GDDR5 | 28 |
| NVIDIA Quadro V100 | Volta | sm_70 | HBM2 | 80 |
| NVIDIA RTX 2060 | Turing | sm_75 | GDDR6 | 30 |

주요 비교는 **Volta QV100 (80 SM)** 기준.

### 6.2 Baselines

| 베이스라인 | 정체 | 설정 방법 |
|---|---|---|
| **GPGPU-Sim 3.x** | 기존 PTX execution-driven 시뮬레이터 | Fermi 구조를 Volta 스펙으로 scaling (SM 수, 메모리 BW 등 단순 조정) |
| **Accel-Sim [PTX Mode]** | 동일 GPGPU-Sim 4.0 perf model, PTX 실행 | Accel-Sim과 ISA만 다른 ablation |

(분석) GPGPU-Sim 3.x는 "best-effort Volta" 설정 — Fermi 시대의 구조로 SM 수 등만 맞춘 것으로, memory system 상세도는 유지되지 않음. 논문 Table VI가 정확한 diff를 보여준다.

### 6.3 워크로드

- **80개 workload (1,945 kernel instances)**: GPGPU-Sim과 동시 실행 가능한 vISA+mISA 공통 워크로드
- **60개 Deepbench workload (11,440 kernel instances)**: Accel-Sim 단독. cuDNN/cuBLAS hand-tuned SASS 포함

CUDA 10 컴파일, input size는 각 benchmark suite 기본값. Deepbench는 학습 10개 + 추론 10개 입력. Hardware counter는 nvprof/nsight-cli로 per-kernel 기준 다회 실행 평균.

### 6.4 메트릭 정의

| 메트릭 | 적용 대상 | 정의 |
|---|---|---|
| MAE | cycles, instructions, IPC | $\frac{1}{N}\sum \left|\frac{\hat{y}-y}{y}\right|$ × 100% |
| NRMSE | 나머지 counter | $\frac{\sqrt{\frac{1}{N}\sum(\hat{y}-y)^2}}{\bar{y}}$ |
| Correl | 모든 counter | Karl Pearson coefficient of variation (σ/μ), HW vs Sim 비교용 |

MAE는 참값이 0인 경우 분모가 0이 되어 대형 오류처럼 보이는 문제 있음 → 해당 counter에는 NRMSE 사용. IPC는 machine instruction count를 분자로 통일 (PTX/SASS ISA 중립).

### 6.5 Ablation 구성

논문이 수행하는 ablation은 두 축:

1. **Frontend 효과**: GPGPU-Sim 3.x (PTX+구 perf model) vs Accel-Sim [PTX Mode] (PTX+새 perf model) vs Accel-Sim (SASS+새 perf model)
2. **Memory system 단계별 효과**: Figure 4 counter별 분해 (L1 req, L1 hit, L2 reads, L2 hits, DRAM reads, IPC, occupancy)

개별 memory feature별 ablation (예: IPOLY만 끄면 얼마나 나빠지는가)은 **논문에 없음**.

### 6.6 🔍 재현 체크리스트

| 항목 | 상태 | 비고 |
|---|---|---|
| 코드 공개 | ✅ | [github.com/accel-sim](https://github.com/accel-sim/accel-sim-framework) |
| Trace 공개 | ⚠️ | 일부 공개, 전체 6.2 TB는 별도 다운로드 |
| Hardware counter 수집 스크립트 | ✅ (논문 언급) | nvprof/nsight-cli |
| Configuration 파일 | ✅ | GitHub에 포함 |
| Deepbench 트레이스 | ⚠️ | cuDNN 버전 고정 필요 |
| CUDA 버전 | ✅ | CUDA 10 명시 |
| Turing 트레이스 | ⚠️ | Volta(sm_70) 트레이스를 Turing(sm_75)에 재사용, NVBit Turing 미지원 |
| 검증 GPU 접근 | ❌ | QV100, TITAN X 등 특정 카드 필요 |
| 개별 feature ablation | ❌ 논문에 없음 | IPOLY, write policy, sector cache 각각의 기여 분해 없음 |
| warp/memory 스케줄러 exact 정책 | ❌ 논문에 없음 | "hardware correlation이 가장 높은 조합" 선택이지만 상세 미공개 |

---

## 7. 결과 분석

### 7.1 전체 cycle 정확도 (Volta QV100 기준)

| 시뮬레이터 | MAE (cycles) | Correl |
|---|---|---|
| GPGPU-Sim 3.x | 94% | 0.87 |
| Accel-Sim [PTX Mode] | 34% | 0.98 |
| **Accel-Sim (SASS)** | **15%** | **1.00** |

PTX Mode → SASS로의 전환만으로 MAE 34% → 15% (약 2×). 나머지 절반의 개선은 perf model 업데이트.

### 7.2 Counter별 상세 비교 (Volta, 논문 Table VII)

| Counter | GPGPU-Sim 3.x | Accel-Sim |
|---|---|---|
| Instructions | MAE=27%, Correl=0.87 | MAE=1%, Correl=1.00 |
| Execution Cycles | MAE=94%, Correl=0.87 | MAE=15%, Correl=1.00 |
| L1 Reqs (post-coalescer) | NRMSE=3.04, Correl=0.99 | NRMSE=0.00, Correl=1.00 |
| L1 Hit Ratio | NRMSE=1.04, Correl=0.69 | NRMSE=0.76, Correl=0.87 |
| L2 Reads | NRMSE=2.67, Correl=0.95 | NRMSE=0.03, Correl=1.00 |
| L2 Read Hits | NRMSE=3.24, Correl=0.82 | NRMSE=0.47, Correl=0.99 |
| DRAM Reads | NRMSE=5.69, Correl=0.89 | NRMSE=0.92, Correl=0.96 |
| Occupancy | NRMSE=0.12, Correl=0.99 | NRMSE=0.13, Correl=0.99 |

**주목할 점**: L1 Reqs에서 NRMSE 3.04→0.00 — 이는 sub-warp coalescer + sectoring으로 access count가 정확히 맞아떨어진 것. L1 Hit Ratio는 여전히 0.87에 그침 — profiler가 sector miss를 line tag hit으로 계산하는 것이 원인.

### 7.3 이득이 어디서 오는가 (분해)

```
GPGPU-Sim 3.x (94% error)
    ↓ perf model 업데이트만
Accel-Sim [PTX Mode] (34% error)  → perf model이 약 60%p 기여
    ↓ SASS frontend 추가
Accel-Sim (15% error)              → SASS ISA가 추가 약 19%p 기여
```

**SASS의 기여가 큰 워크로드**: CUTLASS sgemm (MAE 98%→2%), Rodinia bfs, dwt2d, hotspot. 이들은 instruction scheduling이 ILP에 결정적.

**perf model의 기여가 큰 워크로드**: Polybench (atax, bicg, gesummv, mvt, syrk — GPGPU-Sim 최대 1700% → Accel-Sim 23%). 이들은 L1 cache capacity sensitivity가 높은데 GPGPU-Sim은 32KB L1 고정인 반면 Accel-Sim은 128KB adaptive L1 사용.

**예외**: CUTLASS gemm-wmma에서는 Accel-Sim [PTX Mode]가 MAE≈0.5%로 가장 정확. 추상적 WMMA PTX가 우연히 cycle-accurate에 가까운 것. SASS HMMA 모델이 일부 입력 크기에서 덜 정확.

### 7.4 Bandwidth 달성률 (Volta, Figure 7)

| 수준 | 이론 BW 대비 달성률 |
|---|---|
| **L1 BW** | HW: ~95%, Accel-Sim: 85%, GPGPU-Sim: 33% |
| **L2 BW** | HW: ~95%, Accel-Sim: ~73%, GPGPU-Sim: ~50% |
| **Mem BW** | HW: 85%, Accel-Sim: 82%, GPGPU-Sim: 62% |

L2 BW 22% 잔여 오차: IPOLY hashing이 정확한 하드웨어 hash function을 완벽히 모방하지 못함(논문 인정).

### 7.5 Deepbench 결과 (Figure 6)

Accel-Sim만 시뮬레이션 가능. Correl=0.95, MAE=33%. 오차가 높은 이유: texture memory, local load 모델링 미흡(논문 명시). RNN workload의 낮은 occupancy는 실리콘 GPU 연구와 일치.

### 7.6 Case Studies

**Case Study 1: L1 Cache Throughput 병목**

GPGPU-Sim: srad-v1, sc, mm 등에서 L1 reservation fails (MSHR\_RESERV\_FAIL + LINE\_ALLOC\_FAIL + QUEUE\_FULL) 수백~1000/kilo-cycles.
Accel-Sim: 동일 워크로드에서 reservation fails ≈ 0.

→ L1 throughput 병목을 해결하려는 기존 논문들(cache bypassing, warp throttling)의 전제가 현대 GPU에서 성립하지 않음. Volta의 streaming + banked + adaptive L1이 이미 병목을 해소했기 때문.

**Case Study 2: FR-FCFS Memory Scheduling**

| | GPGPU-Sim | Accel-Sim |
|---|---|---|
| FR-FCFS vs FCFS 성능 향상 | 평균 20% | **평균 2.5×** |

GPGPU-Sim에서는 FR-FCFS가 일부 워크로드에서 효과 없음. Accel-Sim에서는 훨씬 더 큰 개선. 이유: 더 정확한 coalescing + L2 hashing + write policy가 page locality를 더 많이 깨뜨리고, 이를 FR-FCFS가 회복하는 여지가 커진다. HBM의 dual-bus, pseudo-independent access, per-bank refresh와의 상호작용이 memory scheduling을 더 critical하게 만듦.

---

## 8. 비판적 분석

### 강점

- (분석) 트레이스의 ISA-independent 변환 설계가 깔끔하다. performance model과 frontend가 완전히 분리되어 있어, 새 GPU 지원 시 frontend(ISA def + HW def) 교체만으로 기존 perf model 재사용.
- (분석) counter-by-counter validation 방법론이 체계적이다. "cycle이 맞는다"가 아니라 L1 req → L1 hit → L2 read → DRAM read 계층적으로 각 counter가 맞아야 전체가 맞는 접근.
- (분석) microbenchmark 기반 자동 tuning이 사람이 직접 수행하던 하드웨어 역공학의 상당 부분을 자동화.

### 한계

**① trace-driven의 근본적 제약** (사용자의 추가 사항과 직결)

(분석) 트레이스는 특정 입력에서의 실행 경로를 고정(freeze)한다. 따라서:
- **data-dependent control flow**: 입력이 바뀌면 다른 branch가 탐색됨 → 트레이스 무효화
- **data-dependent memory address**: 입력이 바뀌면 메모리 접근 패턴 변화 → 트레이스의 주소 배열 전체가 달라짐
- **global synchronization, atomic**: 실행 결과에 따라 다음 커널 동작이 달라지는 경우 → 트레이스 재생이 부정확

논문 자체도 이를 인정: "Evaluating new designs that rely on the data values stored in registers or memory and global synchronization mechanisms are either not possible or very difficult to study without emulation-based execution-driven simulation." (논문 §I)

**② Markov 예측 기반 라우팅과의 충돌 (사용자 연구 맥락)**

(분석) 사용자의 data-dependent head routing (Markov 예측)은 본질적으로 trace-driven 시뮬레이션과 충돌한다. Markov 예측의 정확도는 과거 routing 결과에 따라 달라지고, routing 결정이 어떤 메모리/compute가 활성화되는지를 결정한다. 이 feedback loop는 트레이스에 포착되어 있지 않다.

구체적 문제:
- 트레이스에 기록된 active_mask와 메모리 주소는 Markov 예측이 "정확했을 때"의 경로를 반영
- 예측 오류(misprediction) 패널티, 올바른/틀린 head 로드에 따른 분기가 트레이스에 없음
- 따라서 Accel-Sim trace-driven 모드로는 "예측 정확도가 X%일 때 실제 cycle이 얼마인가"를 측정할 수 없음

→ **가능한 접근**: execution-driven PTX 모드 사용 + Markov 예측 로직을 PTX emulator에 삽입. 또는 여러 시나리오(완벽 예측 / 완전 실패)에 대해 별도 트레이스를 생성하여 bound를 측정.

**③ 모델링 누락 항목**

- texture memory, local load 상세 모델링 미완 (Deepbench 33% error 원인으로 논문 자체 지적)
- L2 hashing 함수 미완 (22% L2 BW 오차 잔존)
- warp scheduling 정확한 정책: microbenchmark로 결정하지 못하고 bandwidth 상관 최고 조합 선택 → 실제 HW 정책과 다를 가능성
- Turing 트레이스를 Volta trace로 대체 (NVBit Turing 미지원)

**④ 멀티스레딩 미지원**

12.5 KIPS (trace-driven) 속도는 대규모 LLM inference 커널에서 심각한 병목. Deepbench 60개 workload (11,440 kernel instances) 시뮬레이션에 논문 기준 수 주 수준 예상. (외부: arXiv 2502.14691에서 병렬화 시도, determinism 문제 언급)

**⑤ 개별 feature 기여 분해 ablation 없음**

sub-warp coalescer, adaptive L1, IPOLY hashing, write-validate policy 각각이 cycle accuracy에 얼마나 기여하는지 분해 실험이 없다. PTX Mode → SASS Mode로의 전환이 "나머지"를 모두 설명한다고 가정하는 구조.

---

## 9. 관련 연구 맥락 (웹조사)

### 9.1 선행 연구 lineage

| 논문 | 역할 | 관계 |
|---|---|---|
| GPGPU-Sim (Bakhoda et al., ISPASS'09) | 기반 simulator | Accel-Sim이 GPGPU-Sim 4.0으로 확장 |
| Jain et al. SIGMETRICS'18 (외부) | "GPGPU-Sim은 Pascal에서 부정확하다" 실증 | Accel-Sim의 motivation paper |
| Raihan et al. ISPASS'19 (외부) | Volta sub-core, HMMA Tensor Core 모델링 | Accel-Sim이 직접 채용 |
| NVBit (Villa et al., MICRO'19) (외부) | GPU dynamic binary instrumentation | Accel-Sim tracer의 기반 |
| Jia et al. arXiv'18 (외부) | Volta 마이크로벤치마크로 역공학 | Accel-Sim이 microbenchmark 설계에 참고 |
| Rau ISCA'91 (외부) | IPOLY pseudo-random memory interleaving | L2 hashing 직접 인용 |

### 9.2 동시기·후속 연구

- (외부) **AccelWattch (MICRO'21)**: Accel-Sim 기반에 전력 모델 추가, Volta QV100 validated. GPGPU-Sim 4.2.0에 통합.
- (외부) **"Analyzing and Improving Hardware Modeling of Accel-Sim" (arXiv'24)**: instruction cache 모델링 오류(branch 무관 PC packing), control flow 명령어 dependency 누락 등 구체적 버그 지적 및 수정.
- (외부) **GainSight (arXiv'25)**: Accel-Sim을 백엔드로 활용하여 H100 메모리 접근 패턴 프로파일링. Accel-Sim 1.3.0 + GPGPU-Sim v4.2.1 사용.

### 9.3 Novelty 평가

(분석) 논문이 "최초"라고 주장하는 항목:
- NVIDIA SASS 시뮬레이션 가능한 오픈소스 framework: **실질적으로 최초** (동시기 경쟁자 없음)
- cuDNN/cuBLAS 시뮬레이션: **최초** (다른 시뮬레이터는 PTX fallback 없이 불가)
- 4세대 NVIDIA GPU validated 오픈소스: **최초**

sub-warp sectored coalescer, L1 streaming cache 등의 발견은 Raihan et al.'19, Jia et al.'18 등의 역공학 작업 위에 서 있으나, 이를 통합된 validated 시뮬레이터로 구현한 것은 Accel-Sim이 처음.

---

## 10. 🎯⭐ 내 연구와의 연결점 · 적용 가능성

### 10.1 빌려올 수 있는 아이디어/기법

**① Validation 방법론 — counter-by-counter 계층적 검증**

메모리 hierarchy를 L1 req → L1 hit → L2 reads → DRAM reads 순서로 각 계층 counter를 독립 검증하는 접근. 내 offloading/prefetch 실험에서도 동일 패턴 적용 가능:
- prefetch hit rate (L1/L2에서의 early arrival)
- offloading 시 DRAM 접근 패턴 변화
- memory bandwidth 활용률

**② Microbenchmark 기반 Latency/Bandwidth 측정**

pointer-chasing으로 각 memory level의 정확한 latency를 측정하는 패턴은 내 prefetch window size 결정, offloading overhead 측정에 직접 활용 가능.

**③ IPOLY hashing이 L2 BW에 미치는 영향**

MoE/MoH offloading에서 expert weight를 DRAM에서 L2를 통해 로드할 때, access pattern이 $2^k$ stride를 가질 경우 partition camping이 심각해질 수 있음. IPOLY 설정을 활성화하고 Accel-Sim으로 캠핑 완화 효과를 정량화하는 것이 유효.

**④ Simulation-gating**

내가 직접 simulator를 수정할 경우, "active 컴포넌트만 tick"하는 simulation-gating 패턴은 prefetch/offloading 관련 idle period가 많을 때 속도 향상에 유용.

### 10.2 비교 baseline으로 활용

Accel-Sim (GPGPU-Sim 4.0, Volta QV100 config)은 이미 공개·validated. 내 연구의 contribution을 검증하기 위한 cycle-accurate baseline으로 가장 적합한 오픈소스 선택지.

- prefetch 없는 순수 demand-fetch 시나리오의 baseline cycle/memory traffic
- DRAM bandwidth saturation 하에서 FR-FCFS 등 스케줄링 정책의 baseline 효과

### 10.3 Trace-driven vs Execution-driven: 나의 사용 전략

| 시나리오 | 권장 모드 | 이유 |
|---|---|---|
| MoE/MoH expert weight 단순 offload 측정 | **trace-driven** | routing이 고정된 경우, 트레이스가 정확하게 재현 |
| Markov 예측 기반 dynamic routing 성능 | **execution-driven PTX** | routing 결과가 다음 메모리 패턴을 결정 → 고정 트레이스 불가 |
| KV cache prefetch 효과 측정 | **trace-driven + multiple traces** | 서로 다른 prefetch 타이밍 시나리오를 별도 트레이스로 표현 |
| INT4 quantized head prediction의 오류 영향 | **execution-driven PTX** | 예측 오류가 control flow를 바꿀 수 있음 |
| prefetch overhead만 측정 (routing 고정) | **trace-driven** | baseline 대비 DRAM 접근 감소 정량화 가능 |

**핵심 결론**: data-dependent Markov routing은 trace-driven Accel-Sim과 근본적으로 충돌한다. 이 기능을 시뮬레이션하려면 execution-driven PTX 모드에서 Markov 예측 로직을 직접 삽입하거나, 예측 정확도 구간(0%, 50%, 100%)에 따른 multiple trace sets를 생성하여 interpolation하는 간접 접근이 필요하다.

### 10.4 출력 통계 카운터 — 내 평가 지표 후보

| Accel-Sim 카운터 | 내 실험에서의 의미 |
|---|---|
| L2 Reads (NRMSE) | expert weight offloading 시 L2 traffic 변화 |
| DRAM Reads (NRMSE) | prefetch hit/miss에 따른 DRAM bandwidth 변화 |
| L1 Hit Rate | KV cache가 L1에 얼마나 머무는지 |
| L2 Read Hits | prefetch가 L2에 제때 도착했는지 |
| Execution Cycles (MAE) | 전체 inference latency 변화 |
| IPC | compute utilization (memory stall 비율 역산 가능) |
| MSHR 상태 | prefetch inflight request 수, backpressure 측정 |

### 10.5 충돌하거나 재검토가 필요한 가정

- (분석) Accel-Sim의 L2 BW가 22% 과소 추정됨 → L2 BW가 bottleneck이 되는 시나리오에서 DRAM offloading 효과가 실제보다 작게 나올 위험. 실제 HW 실험으로 검증 필요.
- (분석) texture memory 모델링 미완 → cuDNN 기반 attention kernel 시뮬레이션 시 오차 증가 가능.
- (분석) Turing trace를 Volta trace로 대체한 경우 — RTX 계열 GPU에서 실험 시 ISA 차이 무시.

---

## 11. 핵심 용어 정리

| 용어                      | 정의                                                                             |           |                   |
| ----------------------- | ------------------------------------------------------------------------------ | --------- | ----------------- |
| vISA (virtual ISA)      | PTX: NVIDIA의 추상 ISA. 컴파일러가 생성, 드라이버가 mISA로 JIT 컴파일.                            |           |                   |
| mISA (machine ISA)      | SASS: 특정 GPU 세대의 실제 하드웨어 명령어. sm_70=Volta, sm_75=Turing 등.                     |           |                   |
| NVBit                   | NVIDIA의 dynamic binary instrumentation framework. SASS 레벨에서 동작, LD_PRELOAD 방식. |           |                   |
| SASS                    | Shader ASSembly. NVIDIA GPU의 mISA.                                             |           |                   |
| sub-core                | Volta 이후 SM의 기본 단위: warp scheduler + 전용 RF + 전용 EU.                            |           |                   |
| sectored cache          | 128B cache line을 32B sector 4개로 분할. tag는 128B, data fetch는 32B 단위.             |           |                   |
| partition camping       | $2^n$ 메모리 bank에서 $2^k$ stride 접근이 특정 bank에 집중되는 현상.                            |           |                   |
| IPOLY hashing           | Galois field irreducible polynomial 기반 L2 bank index 생성. camping 완화.           |           |                   |
| write-validate          | write miss 시 DRAM fetch 없이 sector에 기록 후 byte-level mask로 유효성 추적.               |           |                   |
| MSHR                    | Miss Status Holding Register. in-flight cache miss 추적 구조.                      |           |                   |
| simulation-gating       | 할 일 없는 컴포넌트의 cycle tick을 skip하는 속도 최적화.                                        |           |                   |
| FR-FCFS                 | First-Row First-Come First-Serve. row-buffer hit를 우선 처리하는 DRAM 스케줄러.           |           |                   |
| active mask             | warp 내 현재 cycle에 active한 thread 비트맵 (32bit).                                   |           |                   |
| ISA Def 파일              | SASS opcode → execution unit 매핑 파일. 사용자가 공개 문서로 작성.                            |           |                   |
| HW Def 파일               | SM 수, warp 크기 등 기초 HW 파라미터 정의 파일. microbenchmark가 참조.                          |           |                   |
| MAE                     | Mean Absolute Error. $\frac{1}{N}\sum                                          | \hat{y}-y | /y \times 100\%$. |
| NRMSE                   | Normalized RMSE. 참값이 0인 counter에 사용.                                           |           |                   |
| Correl                  | Karl Pearson coefficient of variation (σ/μ). HW vs Sim 추세 일치도.                 |           |                   |
| kernelslist             | Accel-Sim trace 디렉토리의 kernel 실행 순서 메타파일.                                       |           |                   |
| base+stride compression | 규칙적 메모리 접근 패턴을 (base, stride) 쌍으로 압축.                                          |           |                   |

---

## 12. 미해결 질문 / 더 읽을 것

### Accel-Sim이 남긴 open questions

1. L2 bandwidth의 22% 잔여 오차 — 실제 HW hash function 완전 역공학 필요
2. warp scheduling / replacement policy의 정확한 모델 — hit rate correlation 개선 필요
3. texture memory / local load 상세 모델링 — Deepbench 33% 오차 해소
4. Turing NVBit 지원 이후 sm_75 전용 트레이스로의 재검증
5. multithreaded simulation 가능성과 determinism 보장 방법

### 더 읽을 논문

- **Raihan et al., "Modeling Deep Learning Accelerator Enabled GPUs," ISPASS'19** — sub-core 모델, HMMA Tensor Core 분석의 원천 [(링크)](https://ieeexplore.ieee.org/document/8695671)
- **Jain et al., "A Quantitative Evaluation of Contemporary GPU Simulation Methodology," SIGMETRICS'18** — GPGPU-Sim Pascal 검증 문제 실증, Accel-Sim motivation [(arXiv)](https://arxiv.org/abs/1710.03090)
- **Villa et al., "NVBit: A Dynamic Binary Instrumentation Framework for NVIDIA GPUs," MICRO'19** — Accel-Sim tracer의 기반 [(ACM DL)](https://dl.acm.org/doi/10.1145/3352460.3358307)
- **Jia et al., "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking," arXiv'18** — Volta L1/L2 역공학 [(arXiv)](https://arxiv.org/abs/1804.06826)
- **Khairy et al., "A Detailed Model for Contemporary GPU Memory Systems," ISPASS'19** — Accel-Sim ISCA'20의 직전 워크샵 버전, memory system 변경 사항만 집중 [(저자 PDF)](https://mkhairy.github.io/Docs/ispass2019.pdf)
- **Huerta et al., "Analyzing and Improving Hardware Modeling of Accel-Sim," arXiv'24** — Accel-Sim의 instruction cache, control flow 모델링 버그 분석·수정 [(arXiv)](https://arxiv.org/abs/2401.10082)
- **AccelWattch (MICRO'21)** — Accel-Sim에 전력 모델 추가 [(accel-sim GitHub)](https://github.com/accel-sim/accel-sim-framework)
