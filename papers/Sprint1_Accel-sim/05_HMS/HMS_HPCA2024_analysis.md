# Bandwidth-Effective DRAM Cache for GPUs with Storage-Class Memory

## 0. 한눈에 보기

- **출처**: HPCA 2024 / Jeongmin Hong, Sungjun Cho, Geonwoo Park, Wonhyuk Yang, Young-Ho Gong, Gwangsun Kim / POSTECH & Soongsil University  
  DOI: [10.1109/HPCA57654.2024.00021](https://doi.org/10.1109/HPCA57654.2024.00021)
- **분류 태그**: `#DRAM-cache` `#SCM` `#PCM` `#GPU-memory` `#HBM` `#HW-SW-codesign` `#memory-oversubscription` `#accelerator-simulation`
- **TL;DR**: GPU에 Phase Change Memory(PCM) 기반 SCM을 붙여 용량 벽을 돌파하되, DRAM cache의 bandwidth overhead를 AMIL(tag 집약), CTC(L2 repurpose tag cache), SCM-aware bypass policy 세 메커니즘으로 제거해 oversubscribed HBM 대비 최대 12.5× (평균 2.9×) 성능 향상을 달성한다.
- **핵심 기여**:
  1. **HMS (Heterogeneous Memory Stack)**: SCM die + DRAM die를 TSV로 3D stacking, 같은 channel에서 shared bus로 유연한 BW 배분
  2. **AMIL (Aggregated Metadata-In-Last-Column)**: DRAM cache row의 모든 tag를 마지막 column의 32 B data portion에 집약 → single column access로 row 전체 tag 취득, ECC 온전히 유지
  3. **SCM-aware DRAM Cache Bypass Policy**: (spatial locality + write intensity → SCM penalty score) × access frequency → DRAM-affinity score의 2-level 비교 구조
  4. **CTC (Configurable Tag Cache)**: L2 cache way의 일부를 tag cache로 repurpose, user-configurable 분할
  5. **SCM power/mode management**: throttling(timing parameter 배가) + SLC/MLC/TLC 모드 전환으로 footprint 크기에 adaptive 대응
- **한 줄 직관**: "SCM은 sequential read엔 HBM만큼 빠르지만 random write엔 치명적이다 — 이 비대칭성을 3차원 score로 정량화해 DRAM cache가 정말 필요한 데이터만 골라 담으면, oversubscribed HBM보다 훨씬 낫다."

---

## 1. 문제 정의 & 동기

### 1-1. 어떤 문제를, 왜 지금

(논문) GPU의 compute throughput은 연평균 1.84×/yr로 성장하는데 memory capacity는 1.39×/yr에 그친다(Fig. 1a). 대규모 graph analytics, LLM training/inference 등 memory-intensive workload에서 HBM 용량 초과는 일상이 됐다.

현재 해법들의 문제:
- **Unified Memory (demand paging)**: page fault handling latency ~20 µs + PCIe BW 12.8 GB/s 병목 → bfs에서 25% oversubscription만으로도 ~4.5× 슬로우다운
- **Multi-GPU**: inter-GPU switch 비용 + pin BW의 sub-linear 스케일링 → memory capacity per GPU cost가 오히려 낮아짐
- **NVLink host 연결**: HBM이 thrash되면 page migration overhead가 여전히 지배

### 1-2. 기존 DRAM cache 접근의 gap

(논문) CPU용 DRAM cache 선행 연구들(BEAR, RedCache, Alloy Cache 등)의 한계:

| 한계 | 내용 |
|------|------|
| BW sensitivity mismatch | CPU는 latency 민감, GPU는 BW 민감 |
| SCM-agnosticism | SCM의 비대칭 read/write 특성 무시 |
| 64 B cacheline | GPU의 inter-thread spatial locality 활용 미흡 |
| ECC 침해 | TAD 방식은 ECC bit 일부를 tag 저장에 전용 |
| Static tag cache | 용량 고정 → workload 적응성 없음 |

---

## 2. 핵심 아이디어

SCM은 **row buffer hit 상태의 sequential read는 DRAM과 동등**하지만 **random write의 write recovery(tWR=1000 cyc)는 DRAM(16 cyc)의 62.5×**에 달한다. 이 비대칭성을 "SCM penalty score"(현재 접근 패턴의 DRAM 대비 latency overhead per column access)와 "DRAM-affinity score"(score × hotness)로 수치화해, GPU의 100,000개 thread가 만들어내는 접근 패턴을 **single-dimensional score**로 압축한다. Score가 낮은 데이터(locality 높고 read-heavy)는 SCM에서 직접 서비스하고, score가 높은 데이터(random write 집중)만 DRAM cache에 fill한다. 여기에 tag probe BW overhead는 AMIL+CTC로, SCM write traffic은 bypass policy로 동시에 제거한다.

---

## 3. 배경 (필요한 것만, 압축)

### 3-1. SCM(PCM) 타이밍 비대칭성

(논문) 핵심 수치 — HBM organization 기준:

| 파라미터 | DRAM | SCM(SLC) | SCM(MLC) |
|---------|------|----------|----------|
| tCL (cyc) | 14 | 14 | 14 |
| tRCD (cyc) | 14 | 120 | ~250 |
| tWR (cyc) | 16 | 1000 | ~2350 |
| tRP (cyc) | 14 | 14 | 14 |
| Row hit latency | 15 ns | 15 ns | 15 ns |
| Row miss latency | 43 ns | 149 ns | ~280 ns |

**핵심**: tCL은 같다 → row buffer locality가 높으면 SCM도 channel BW 포화 가능. 그러나 tWR 차이로 write traffic은 반드시 DRAM cache에서 필터링해야 한다.

### 3-2. HMS의 channel sharing 이점

(논문) DRAM과 SCM을 **같은 channel에 separate rank로** 배치(Fig. 6a). 이렇게 하면 DRAM cache hit rate가 변해도 channel BW를 100% 활용 가능. 반면 separate channel(Fig. 6b)은 hit rate 변화에 따라 한쪽 channel이 놀게 된다(최악 50% utilization).

### 3-3. UVM simulation 방법론

(외부, [46] Ganguly et al. ISCA'19) GPGPU-sim UVMSmart: GPGPU-Sim v3.x에 cudaMallocManaged, page fault handling, TBN prefetcher, pre-eviction policy를 추가한 simulator. (논문) 이 논문은 이를 Accel-sim + Ramulator와 통합하여 사용.

---

## 4. 제안 방법 상세

### 4-1. HMS 구조

```
[GPU die]
   │ PCIe (host)
[Base die] ← I/O buffers, TSV 연결
   │ TSV
[DRAM dies × 4] ← rank 0 (DRAM cache, not programmer-addressable)
   │ TSV
[SCM dies × 4]  ← rank 1 (main memory, 4× DRAM capacity per die)
```

- (논문) SCM이 4× bit density를 가정하면 HMS는 HBM 대비 **2× addressable capacity**
- DRAM cache는 direct-mapped, **256 B cacheline** (CPU용 DRAM cache의 64 B 대비 4×)
- L1/L2는 여전히 128 B line, 32 B sector 유지

### 4-2. AMIL (Aggregated Metadata-In-Last-Column)

**문제**: Alloy Cache(TAD) 방식은 64 B column에 tag+data를 같이 저장 → row 전체를 읽어야 모든 tag 취득. Loh-Hill cache는 앞 몇 column에 tag 집중 → multiple column access 필요. 둘 다 ECC bit을 tag 저장에 유용.

**AMIL의 접근**:

```
DRAM row (2 KiB = 8 × 256B cachelines)
┌──────────────────────────────────────────────────────┬──────────┐
│  cacheline 0  │  cacheline 1  │ ... │  cacheline 6  │ METADATA │
│    (256 B)    │    (256 B)    │     │    (256 B)    │  (32 B)  │
└──────────────────────────────────────────────────────┴──────────┘
                                                          ↑
                          마지막 column: tag + valid + dirty + DRAM-affinity bits
```

**Tag 크기 계산** (논문):
- SCM 4× DRAM → direct-mapped → tag = **2 bits**
- valid 1 bit + dirty 1 bit + DRAM-affinity 2 bits = 4 bits per cacheline
- row당 8 cachelines × 6 bits = **48 bits** 총 metadata
- 32 B column에 48 bits 수납 가능 (여유 있음), ECC 보호도 가능

**비용**: 마지막 column은 SCM data caching 불가 → 1.56% of row는 항상 bypass → 실측 성능 손실 **1.7%**

### 4-3. SCM-aware DRAM Cache Bypass Policy

#### Level 1: SCM Penalty Score

$$\text{SCMPenaltyScore} = \frac{\text{Latency}_\text{SCM} - \text{Latency}_\text{DRAM}}{\text{NumColumnsAccessed}}$$

분자 계산의 핵심 단순화 (논문):
- tCL은 동일하므로 상쇄
- read-only 접근: 분자 ≈ $t_{RCD,SCM} - t_{RCD,DRAM}$
- write 포함 접근: 분자 ≈ $(t_{RCD,SCM} - t_{RCD,DRAM}) + (t_{WR,SCM} - t_{WR,DRAM})$

분모는 MSHR에서 추적하는 column access count.

**구현 비용**: 32-bit register 2개 + ALU 1개 (pre-computed numerator 저장용)

예시 (논문, Fig. 8):
- 4 read hits on same row: score = 93 cyc / 4 = **23 cyc/access** → bypass
- 1 write, no locality: score = 955 cyc / 1 = **955 cyc/access** → cache in DRAM

#### Level 2: DRAM-Affinity Score

$$\text{DRAMAffinityScore} = \text{SCMPenaltyScore} \times \text{PageActivationCounter}$$

- Activation counter: 2 MiB 페이지 granularity, 8-bit, 160 GiB 메모리 기준 80 KiB 오버헤드
- $N_{levels}$ 레벨로 discretize (기본값 4) → 작은 fluctuation에 의한 불일치 방지
- DRAM cache metadata에 2-bit으로 저장 (AMIL의 metadata column에 포함)

#### 2-Level 비교 알고리즘

```
DRAM cache miss 발생:
  1. SCM penalty score 계산 → N_levels 레벨로 discretize
  2. Channel 평균 penalty level과 비교:
     ├─ score ≤ avg: → BYPASS (DRAM 접근 없음, 88.1%의 bypass가 여기서 결정)
     └─ score > avg: → 3단계 진행
  3. victim의 DRAM-affinity level을 DRAM에서 읽음
  4. 현재 요청의 affinity level vs. victim affinity level 비교:
     ├─ 현재 > victim: → REPLACE (victim 퇴출, fill 수행)
     └─ 현재 ≤ victim: → BYPASS (victim 유지)
        * victim의 level을 p_dec 확률로 감소 (p_dec = 현재 activation / max activation)
```

**설계 근거**: 2단계로 분리한 이유는 매 miss마다 DRAM에서 victim affinity를 읽으면 BW overhead가 과도하기 때문. 1단계(SCM penalty 비교)가 88.1%를 필터링해 DRAM 조회를 최소화.

### 4-4. CTC (Configurable Tag Cache)

**구조**:
```
L2 cache (128 B line, 32 B sectors, 16 ways per set)
┌────────────────────────────────────────────────────┐
│  L2 way 0 ~ 11 (고정, L2 cache로 사용)            │
├────────────────────────────────────────────────────┤
│  L2 way 12 ~ 15 (선택적: L2 또는 TC로 사용)       │
│    각 L2 way → 4개 TC way (32 B → 4 × 8 B 섹터) │
│    각 TC sector = 4 B (한 DRAM row의 모든 tag)    │
│    한 TC line = 8개 4B 섹터 = 8개 row의 tag 저장 │
└────────────────────────────────────────────────────┘
```

- **오버헤드**: set당 38 bits(valid/dirty/tag) + 4-bit pseudo-LRU = 612 bits → L2의 **2.5% 스토리지 증가**
- HMS 평가에서 전체 DRAM cache tag의 1/4만 CTC에 수용 (보수적 설정)
- CTC miss 시 AMIL 덕분에 single DRAM access로 row 전체 tag fetch → CTC에 적재

**AMIL과 CTC의 시너지**: TAD 방식이면 CTC miss 시 row 전체(8개 column)를 읽어야 하지만, AMIL은 마지막 column 1회 접근으로 8개 cacheline의 tag를 모두 가져옴.

### 4-5. Power Management

**SCM Throttling**: temperature monitor → tRCD 또는 tWR을 2배로 늘려 power 제한. 평가에서 stencil 외 대부분 workload에서 throttling 불필요.

**SLC/MLC 모드 전환**: footprint이 작으면 DRAM을 memory의 일부로 사용 + SCM SLC 모드(tRCD=60, tWR=150) → HBM 수준의 성능 유지. footprint 증가 시 SCM MLC/TLC 모드로 전환해 용량 확대.

---

## 5. 시스템 · HW 측면

### 5-1. 전체 메모리 접근 흐름

```
L2 cache miss 발생
  │
  ├─ CTC lookup
  │   ├─ HIT: DRAM cache hit/miss 즉시 판단
  │   │      ├─ DRAM cache HIT: DRAM에서 data fetch, SCM penalty avg 업데이트
  │   │      └─ DRAM cache MISS: SCM에서 demand data fetch
  │   │                          → bypass policy 결정
  │   │                          → fill 필요 시 DRAM에 cacheline write
  │   │
  │   └─ MISS: DRAM probe (AMIL: 마지막 column 1 access로 row 전체 tag fetch)
  │            CTC에 tag 적재
  │            이후 위와 동일 흐름
  │
Data 이동 granularity:
  - L2 ↔ DRAM/SCM: 32 B (sector 단위)
  - DRAM ↔ SCM (evict/fill): 256 B (cacheline 단위)
```

### 5-2. HW 구현 범위

| 컴포넌트 | 위치 | 크기 | 역할 |
|---------|------|------|------|
| MSHR per channel | memory controller 근방 | 128 entries × 51 bits | DRAM cache miss 추적 |
| Bypass logic | memory controller | 256-bit storage | score 레지스터 + ALU |
| CTC modification | L2 cache | 38 bits/line + 4-bit pLRU | tag cache 메타데이터 |
| FPU (affinity 계산) | per channel | 0.022 mm² @12nm | 6개 32-bit 레지스터 |
| 총 overhead | A100 40 channel | **1.46 mm²** (0.18%) | — |

(논문) 12nm 기준 면적 추정, A100은 실제 7nm이므로 이 수치는 conservative.

### 5-3. Simulator 구성 (RTL 없음, 아키텍처 레벨 simulation)

(논문) Accel-sim(ISCA'20) + UVM model(Ganguly ISCA'19) + Ramulator(CAL'15)의 통합. Cycle-accurate performance model이나 RTL synthesis는 없음. 열 모델은 HotSpot 6.0 별도 실행.

---

## 6. 🎯⭐ 실험 셋업 & 재현 관점

### 6-1. Simulated System Configuration

(논문, Table I)

| 항목 | 값 |
|------|-----|
| SM 수 | 21 (A100의 1/5 downscale) |
| Warps/SM | 64 |
| Registers/SM | 65,536 |
| Clock | 901 MHz |
| L1+shared | 192 KiB/SM |
| L2 (baseline HBM) | 8 MiB, 16 ways, 128 B line (32 B sectors) |
| L2 (HMS) | 6 MiB, 12 ways (4 ways를 CTC에 할당) |
| CTC | 최대 2 MiB, 32 B line, 4 B sectors |
| L2 latency | 120 cycles, peak BW 402 GB/s |
| Memory channels | 8 channels, 8 dies |
| Row buffer | 2 KiB |
| Bus width | 128-bit (BL2, DDR) |
| Bus frequency | 1 GHz, peak BW 256 GB/s |

타이밍 파라미터:

| | tCL | tRCD | tRAS | tWR | tRP |
|--|-----|------|------|-----|-----|
| DRAM | 14 | 14 | 33 | 16 | 14 |
| SCM | 14 | 120 | 120 | 1000 | 14 |

PCIe BW: 12.8 GB/s (1/5 of PCIe 4.0 ×16)

### 6-2. 🎯 RHBM — "GPU 상주 비율" 손잡이

(논문) `RHBM`은 **HBM이 workload footprint의 몇 %를 담을 수 있는지**를 나타낸다.

기본값: `RHBM = 75%` → HBM은 footprint의 75%만 수용, 25%는 host에 있음.

```
예시: 100 MiB workload, RHBM=75%
  - HBM capacity: 75 MiB
  - HMS DRAM cache: 37.5 MiB (DRAM는 HBM과 같은 용량 per die)
  - HMS SCM: 150 MiB (4× DRAM per die)
  → HMS는 oversubscription 없이 전체 footprint 수용 가능
```

구현 방식: **page frame 수를 `RHBM × footprint`로 제한**. 초과 page는 host에 남겨 UVM이 demand paging으로 처리. 코드 수준에서는 `gpu_memory_allocator`의 가용 page frame 상한값을 조정하는 방식.

> **내 연구와의 매핑**: RHBM = "GPU에 상주하는 head 수 / 전체 head 수". 손잡이 조정 원리는 동일하나, 이 논문은 4 KiB page granularity이고 내 연구는 수십 MB 단위 attention head granularity.

### 6-3. 🎯 cudaMalloc 트레이싱 + 페이지테이블 생성 파이프라인

(논문, §II-B 및 Ganguly ISCA'19 기반, 분석)

시뮬레이터 내부의 UVM 처리 흐름:

```
1. CUDA 앱이 cudaMallocManaged() 호출
   → Accel-sim functional model이 intercept
   → 가상 주소 공간 할당 (host + device 통합 VA)
   → page table entry 초기화 (초기 위치: host memory)

2. GPU kernel 실행 중 memory access 발생
   → L2 cache miss → memory controller로 전달
   → memory controller가 page table lookup
   → page가 GPU(HBM/SCM)에 있으면: 해당 rank로 request 전달
   → page가 host에 있으면: page fault 발생

3. Page fault handling (UVM model, Ganguly):
   → GPU fault queue에 fault address 적재
   → 20 µs 고정 지연 (host driver처리 모델링)
   → PCIe를 통해 4 KiB~1 MiB 데이터 전송 (TBN prefetcher가 크기 결정)
   → page table update (host → GPU)
   → GPU의 해당 page frame에 데이터 적재

4. HMS의 경우:
   → GPU에 적재된 데이터는 SCM rank에 먼저 들어감
   → L2 miss 발생 시 CTC/DRAM cache 조회 후 hit/miss 경로 분기
   → DRAM cache miss: SCM에서 데이터 fetch + bypass policy 결정
```

**논문에 명시 없음**: cudaMalloc 트레이싱의 정확한 코드 구현 위치. UVMSmart GitHub (외부) 기준으로는 `gpgpu_sim_UVMSmart/src/cuda-sim/cuda_runtime_api.cc`에서 `cudaMallocManaged` hook이 구현되어 있으나, Accel-sim과의 통합 버전 코드는 미공개.

### 6-4. 🎯 "느린 계층 접근 = 높은 지연 + prefetch" 모델링 위치

(논문, 분석) 정확한 모델링 위치:

**SCM 지연**: Ramulator의 timing constraint(`tRCD=120 cyc, tWR=1000 cyc`)가 SCM row access 지연을 생성. Accel-sim의 memory request queue → DRAM cache controller → Ramulator SCM rank 순으로 전달되어 해당 timing이 적용됨.

**Page fault 지연 (UVM)**: UVMSmart model에서 page fault event 발생 시 `20 µs` 고정 지연 주입. PCIe BW(12.8 GB/s, 1/5 scale)로 데이터 이동 시뮬레이션.

**TBN Prefetcher**: UVM model 내장. Page fault 발생 시 access pattern을 tree 구조로 분석해 4 KiB~1 MiB adaptive granularity로 prefetch 발행. prefetch도 PCIe를 통해 동일 latency 경로로 처리되나, demand access와 overlap 가능.

**DRAM cache prefetch (fill)**: DRAM cache miss → SCM demand access 완료 → bypass policy가 fill 결정 → DRAM에 256 B cacheline write. 이것이 사실상 SCM→DRAM의 내부 prefetch이며 Accel-sim의 memory controller가 fill request를 Ramulator DRAM rank로 발행.

요약:
| 느린 계층 지연 | 모델링 위치 |
|--------------|-----------|
| SCM row miss (~149 ns) | Ramulator timing model |
| SCM write recovery (~1000 cyc) | Ramulator tWR parameter |
| UVM page fault (~20 µs) | UVMSmart page fault handler |
| PCIe data movement | UVMSmart BW model (12.8 GB/s) |
| TBN prefetch 효과 | UVMSmart prefetcher module |
| DRAM cache fill from SCM | Accel-sim memory controller + Ramulator |

### 6-5. Workloads

(논문) 22개 workload, footprint 19~135 MiB (평균 68 MiB):

| Suite | 워크로드 예시 | 특성 |
|-------|------------|------|
| Pannotia [29] | bfs, gc_*, sssp_*, kcore, cn_cmp | Irregular graph, unpredictable access |
| Rodinia [30] | hsp3D, stencil, pathfnd, bckprp | Regular stencil/iterative |
| Parboil [129] | 2DConv, lavamd | Regular compute-bound |

추가 평가: BERT inference (24.16B params, 480 layers), GPT-3 XL/2.7B training (TensorFlow XLA v2.4, SQuAD)

### 6-6. 🎯 Baselines 상세

| Baseline | 설명 | 출처 |
|---------|------|------|
| HBM | Oversubscribed HBM (RHBM=75%), PCIe UVM | 논문의 primary baseline |
| Infinite HBM (InfHBM) | Never oversubscribed HBM (성능 upper bound) | 논문 |
| SCM | SCM-only 3D stack, DRAM cache 없음 | 논문 |
| BEARi | BEAR (ISCA'15)를 HMS에 적용, **ideal**: DRAM cache presence bit를 이미 안다고 가정, 704 B/channel Neighboring Tag Cache | (외부) Chou, Jaleel, Qureshi, ISCA'15 |
| RedCachei | RedCache (DAC'20), **ideal**: gamma update overhead 없음 | (외부) Behnam & Bojnordi, DAC'20 |
| McCachei | Mostly-clean DRAM cache (MICRO'12), **ideal**: perfect predictor + zero-cost tag probes | (외부) Sim et al., MICRO'12 |
| HMS-B-C (HMS-BP-CTC) | bypass도 CTC도 없는 HMS | ablation |
| HMS-B (HMS-BP) | bypass 없는 HMS (CTC만 있음) | ablation |
| HBM(NVLink) | NVLink host connectivity, Grace Hopper 비율 | 논문 |
| HMS(NVLink) | HMS + NVLink | 논문 |
| Sep. DRAM&SCM | 분리 버스 DRAM+SCM | 논문 design space |
| CXL SCM | CXL 인터페이스 SCM | 논문 design space |

> ⚠️ **중요**: BEAR, RedCache, McCachei 모두 **idealized** 버전(suffix `i`)으로만 평가함. 실제 구현이면 오버헤드가 더 크므로 HMS의 우위는 보수적으로 제시된 것이다.

### 6-7. Hyperparameters

(논문) HMS 핵심 파라미터:

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| Fupdate | 100 | moving average 업데이트 주기 |
| Nlevels | 4 | score discretization 레벨 수 |
| moving avg weight | 1% | 새 값의 가중치 |
| DRAM cacheline | 256 B | — |
| L2/L1 cacheline | 128 B / 32 B sector | — |
| CTC ways | 4 (L2의 1/4) | 보수적 평가 |
| Activation counter | disabled | ideal counter의 speedup은 최대 7.6% (평균 0.4%) |
| pdec | activation count / max activation count | victim affinity 감소 확률 |
| SCM capacity ratio | 4× DRAM per die | HMS addressing 가정 |

### 6-8. 재현 체크리스트

| 항목 | 상태 | 비고 |
|------|------|------|
| Accel-sim 소스 | ✅ 공개 | (외부) accel-sim.github.io |
| UVMSmart 소스 | ✅ 공개 | (외부) DebashisGanguly/gpgpu-sim_UVMSmart |
| Ramulator 소스 | ✅ 공개 | (외부) CMU-SAFARI/ramulator |
| HMS 통합 패치 | ❌ **미공개** | 저자 contact 필요 |
| SCM timing model | ✅ Table I 명시 | Ramulator에 커스텀 추가 필요 |
| Bypass policy 구현 | 📝 상세 서술 | 코드 없음, 재구현 가능 |
| AMIL 구현 | 📝 상세 서술 | Ramulator memory model 수정 필요 |
| Workload traces | 부분 공개 | Pannotia, Rodinia, Parboil 개별 공개 |
| Energy model | ✅ AccelWattch + Table I | 파라미터 명시 |
| Thermal model | ✅ HotSpot 6.0 + Table II | 파라미터 명시 |
| Full A100 simulation | ❌ **불가** | "57년 소요" 추정, 1/5 downscale만 가능 |

**재현상 주요 미보고 항목**:
- DRAM cache controller의 MSHR 우선순위 스케줄링 방식
- moving average의 초기값 설정
- DRAM affinity level 초기화 정책
- CTC miss 시 DRAM probe와 demand access 간의 서비스 순서

---

## 7. 결과 분석

### 7-1. 핵심 성능 수치 (normalized to Infinite HBM, 높을수록 느림)

(논문, Fig. 11)

| 설계 | Overall gmean (vs. InfHBM) |
|------|--------------------------|
| Infinite HBM | 1.000 |
| **HMS** | **1.113** |
| SCM (no DRAM cache) | 1.248 |
| HMS-B (bypass 없음) | 1.281 |
| HMS-B-C (bypass+CTC 없음) | 1.339 |
| HBM (oversubscribed, PCIe) | 1.615 |
| SCM-only (worst) | 3.222 |

HMS는 oversubscribed HBM보다 **2.9× 빠름** (1.615/1.113 ≈ 1.45; 절대 speedup은 그 역수 기준 최대 12.5×).

### 7-2. 이득의 분해 (Traffic 관점)

HMS의 총 memory traffic은 InfHBM 대비 **1.23×** (vs. BEARi 2.06×, RedCachei 2.93×):

```
이득의 원천:
  1. 전체 footprint 수용: page fault 제거 → Host↔GPU copy traffic 소멸
  2. Bypass policy: DRAM write traffic 5.5× 감소, SCM write traffic 3.2× 감소
  3. CTC: DRAM probe traffic 91~93% 감소
  4. AMIL: CTC miss 시 single column access (TAD 대비 8× 적은 DRAM access)
```

### 7-3. Bypass 기여 분해

(논문, Fig. 14)

| 결정 유형 | 비율 | DRAM BW 소모 |
|---------|------|------------|
| 1단계 bypass (SCM penalty ≤ avg) | 88.1% | 없음 |
| 2단계 bypass (victim affinity 비교) | 8.3% | victim affinity 1 read |
| Fill 수행 | 3.6% | 256 B cacheline write |

2단계 비교를 비활성화(HMS-B)하면 stencil에서 최대 49% runtime 증가, 평균 4.8% 증가.

### 7-4. CTC 기여

(논문)

| 메트릭 | 값 |
|--------|-----|
| CTC hit rate | 91% overall (최소 59%) |
| HMS-B vs. HMS-B-C speedup | 최대 40%, 평균 3.9% |
| DRAM probe 감소 vs. BEARi | **93.1%** |
| DRAM probe 감소 vs. RedCachei | **90.6%** |
| L2 miss rate 증가 (4 way CTC 할당 비용) | 5.4% |
| 전체 memory traffic (vs. BEARi) | **40.5% 낮음** |

### 7-5. Cacheline Size 영향

(논문, Fig. 16b) 64 B → 256 B 전환:

| 설계 | Speedup |
|------|---------|
| HMS | +4% |
| HMST (HMS + TAD 대신 AMIL) | +19.4% |
| BEARi | +0.9% |
| RedCachei | 오히려 degradation |

### 7-6. 메모리 용량 vs. 성능 (Fig. 17a)

| Footprint / HBM 용량 | HMS speedup over HBM |
|---------------------|---------------------|
| 1.0 (동일) | 0.996× (거의 같음, SLC 모드) |
| 1.3 | 0.993× |
| 2.0 | 2.894× (MLC 모드) |
| 3.0 | 5.063× |
| 4.0 | 2.85× (HMS도 oversubscribed) |

### 7-7. Energy 및 Thermal

(논문)
- HBM 대비 HMS energy: 최대 89.3% 감소, **평균 48.1% 감소**
- SCM 대비 HMS energy: 평균 16.5% 감소

| 설계 | Peak Temp | 평균 Temp |
|------|----------|----------|
| InfHBM | 361.1 K | 346.0 K |
| **HMS** | **361.4 K** | **344.7 K** |
| BEARi | 362.3 K | 346.3 K |
| RedCachei | **368.5 K** ⚠️ | 352.7 K |

RedCachei는 95°C critical temperature(368 K) 초과.

---

## 8. 비판적 분석

### 강점

(분석) SCM 비대칭성(read-symmetric, write-asymmetric)을 **runtime에 계산 가능한 단일 수치**로 압축하는 SCM penalty score의 아이디어가 핵심이다. 별도 profiling phase 없이 timing parameter와 MSHR에서 추출한 column access count만으로 계산할 수 있어 overhead가 극히 낮다. CPU 연구의 "재사용 빈도 기반" bypass와 GPU의 "spatial locality 기반" bypass를 DRAM-affinity score로 통합한 것도 우아하다.

(분석) AMIL의 "마지막 column을 metadata로 희생"이라는 아이디어는 단순하지만 효과적이다. 1.6% 희생으로 tag access BW를 8× 줄이고 ECC도 유지한다. 이런 asymmetric resource 활용 아이디어는 다른 메모리 계층 설계에도 이전 가능하다.

### 한계 및 숨은 가정

(분석)

**1. SCM 모델의 낙관성**: 논문은 on-die raw PCM을 가정하지만, 현재 상용 SCM(Optane)은 on-DIMM buffer + 내부 큐가 있어 raw PCM 특성이 나타나지 않는다. 논문도 §II-D에서 이를 인정한다. TSV 적층 raw PCM이 상용화되지 않은 현재, 이 연구는 미래 기술 가정 위에 있다.

**2. Downscaling 타당성**: A100의 1/5 downscale(21 SM, 6~8 MiB L2)에서의 결과가 실제 108 SM, 40 MiB L2에서도 동일하다는 보장이 없다. L2:memory 비율이 달라지면 CTC hit rate와 bypass 비율이 변한다.

**3. Channel-level 평균의 한계**: SCM penalty score를 channel 평균과 비교하는 방식은 SM 간 access pattern heterogeneity를 무시한다. 서로 다른 kernel이 동시에 실행될 때 평균이 왜곡될 수 있다.

**4. Write endurance 미평가**: DRAM cache의 write filtering이 PCM 수명에 얼마나 기여하는지, 그리고 write-heavy workload에서 SCM이 먼저 마모되는지 미분석.

**5. Idealized baseline**: BEARi/RedCachei/McCachei가 모두 이상적 버전이므로 HMS의 우위가 보수적으로 제시됐다. 실제 구현 버전과 비교하면 HMS가 더 유리할 것이다.

**6. TSV bus 경합 미상세화**: DRAM rank와 SCM rank가 같은 bus를 공유할 때 DRAM cache fill request와 SCM demand access 간의 경합 및 우선순위 처리가 명시되지 않음.

### 일반화 가능성

(분석) 이 논문의 아이디어는 "DRAM cache + 느린 backing store + GPU"의 조합이면 어디에나 적용 가능하다. Flash(ZnG, FlashGPU), CXL-attached DRAM, NVLink remote memory 등 다양한 backing store에 bypass policy와 AMIL 아이디어를 이전할 수 있다. 단, SCM penalty score의 분자 계산은 해당 backing store의 timing 비대칭성에 맞게 재정의해야 한다.

---

## 9. 관련 연구 맥락 (웹조사)

### Prior Work Lineage

```
[DRAM cache tag 관리 계보]
Loh-Hill Cache (IEEE Micro'12) [95]
  → 29-way associative, tags in first few columns
  → 문제: multiple column access for full row tag fetch

Alloy Cache / TAD (MICRO'12) [113] ← Qureshi & Loh
  → direct-mapped, single TAD access
  → 문제: ECC bit repurposing, row-level tag scatter → CTC miss시 row 전체 접근

BEAR (ISCA'15) [35] ← Chou, Jaleel, Qureshi
  → probabilistic bypassing + LLC-based presence bit + neighbor tag cache
  → 대상: CPU, HBM-backed off-package DRAM cache
  → 한계: SCM-aware 아님

RedCache (DAC'20) [25]
  → access count threshold bypass
  → 한계: SCM-aware 아님, low-count page 전부 bypass → SCM write 폭증

McCachei (MICRO'12) [124]
  → mostly-clean DRAM cache
  → 한계: SCM write energy 완전 무시

HMS (HPCA'24) ← 본 논문
  → AMIL + CTC + SCM-aware 2-level bypass
  → GPU inter-thread spatial locality 최초 반영
```

### Simulator Lineage (외부 조사)

- **GPGPU-Sim** (Bakhoda et al., ISPASS'09): 최초 상세 GPU simulator
- **UVMSmart** (Ganguly et al., ISCA'19): GPGPU-Sim 3.x에 UVM + page fault + TBN prefetcher 추가. cudaMallocManaged, cudaMemPrefetchAsync 지원
- **Accel-sim** (Khairy et al., ISCA'20): SASS trace-driven frontend + GPGPU-Sim 4.0 performance model. 복수 GPU generation 지원, NVBit 기반 trace
- **AccelWattch** (Kandiah et al., MICRO'21): Accel-sim 기반 power model

(분석) 본 논문은 **Accel-sim + UVMSmart + Ramulator**의 3-way 통합으로, GPU microarchitecture + UM page migration + detailed DRAM/SCM timing을 동시에 시뮬레이션하는 현존하는 가장 comprehensive한 GPU memory research infrastructure다.

### Novelty의 실제 위치

(분석) CPU 쪽의 DRAM+PCM hybrid 연구(Qureshi et al. ISCA'09, Yoon et al. ICCD'12)는 2009년부터 존재했지만, GPU의 warp-level inter-thread spatial locality, 100,000+ thread concurrency, BW-first 특성을 모두 고려한 DRAM cache 설계는 이 논문이 최초다. "GPU용 SCM+DRAM cache design space의 최초 탐구"라는 claim은 타당하다.

---

## 10. 🎯⭐ 내 연구와의 연결점 · 적용 가능성

### 10-1. 🎯 Simulator Fork 베이스 적합성 평가

**결론: 이 논문의 simulator stack이 내 연구의 가장 근접한 베이스이지만, 코드가 미공개이므로 개별 컴포넌트를 재조합해야 한다.**

| 요소 | 이 논문 | 내 연구 |
|------|---------|---------|
| Base simulator | Accel-sim | Accel-sim (동일) |
| Memory model | Ramulator (DRAM+SCM timing) | HBM + CPU DRAM (PCIe/NVLink 지연) |
| 2계층 구조 | DRAM cache (HBM) + SCM (on-device) | GPU HBM + CPU DRAM (off-device) |
| 상주 비율 손잡이 | RHBM (page frame 수 조정) | R_GPU (상주 head 수 조정) |
| 느린 계층 지연 | SCM row miss ~149 ns | PCIe ~µs / NVLink ~0.135 µs |

**필요한 수정사항**:
1. Ramulator의 SCM rank → CPU DRAM 접근 모델 (PCIe/NVLink latency)로 교체
2. DRAM cache HW logic(AMIL, CTC, bypass policy) 코드 재구현 (미공개)
3. 4 KiB page granularity UVM → attention head 단위 (~수십 MB) offloading 정책으로 교체
4. Head-level prefetch trigger 및 prefetch buffer 모델 추가

### 10-2. 🎯 HBM+SCM 2계층 표현 방식 (설계 참조)

(논문, 분석) 2계층의 핵심 표현 구조:

**주소 공간 및 페이지 관리**:
- SCM만 programmer-addressable (DRAM cache는 HW-managed, 투명)
- UVM page table은 SCM 주소 기준
- DRAM cache는 direct-mapped: `SCM_address % DRAM_cache_size → cache_set`
- Page fault 처리 → SCM에 4 KiB 적재 → 이후 DRAM cache가 256 B cacheline 단위로 hot data caching

**지연 계층 모델링**:
```
L2 miss → CTC 조회 (SRAM, ~0 추가 지연)
  → DRAM cache hit: +15~43 ns (DRAM row hit/miss)
  → DRAM cache miss → SCM access: +15~149 ns (SCM row hit/miss)
  → UVM page fault (SCM에도 없음): +20 µs + PCIe copy
```

**내 연구 매핑**:
```
L2 miss → head 배치 테이블 조회 (SRAM)
  → GPU HBM hit: +HBM latency
  → CPU DRAM (offloaded head): +NVLink/PCIe latency (~0.135 µs ~ µs)
```

### 10-3. 🎯 RHBM = 내 R_GPU 손잡이

(논문의 RHBM 구현 방식):
```
# 시뮬레이터 레벨:
gpu_page_frames = (total_workload_footprint_pages) * RHBM
# RHBM = 0.75 → 75%의 page frame만 GPU에 할당
# 초과 page = host에 유지 → demand paging으로 서비스
```

내 연구로의 매핑:
```
# head-level:
gpu_heads = total_heads * R_GPU
# R_GPU = 0.5 → 50%의 head만 GPU HBM에 상주
# 나머지 = CPU DRAM → prefetch + demand access
```

**Fig. 17b 실험 구조 활용**: RHBM을 25%→50%→75%→100%로 sweep하며 성능 곡선 생성. 내 연구에서도 R_GPU를 10%~100%로 sweep해 "최적 상주 head 수 vs. prefetch accuracy" tradeoff 곡선 생성 가능.

### 10-4. SCM Penalty Score 아이디어의 이전 가능성

(분석) 내 연구에서의 "head offloading utility score" 설계에 활용 가능:

$$\text{HeadOffloadScore} = \frac{\text{AccessFrequency} \times \text{Reuse}}{\text{PCIe\_transfer\_cost}}$$

- AccessFrequency: attention 연산에서 head별 접근 빈도 (MoH에서 head별 routing 통계)
- Reuse: decode step 간 같은 head 재접근 빈도 (KV cache reuse distance)
- PCIe_transfer_cost: head 크기 × PCIe latency

Score가 낮은 head → CPU에 offload; 높은 head → GPU 상주. 이 구조는 이 논문의 bypass policy와 isomorphic하다.

### 10-5. 구체적 적용 시나리오

| 이 논문의 기법 | 내 연구 적용 방식 |
|--------------|----------------|
| RHBM sweep (Fig. 17) | R_GPU sweep으로 "상주 head 수 vs. throughput" 곡선 생성 |
| 2-level bypass의 1단계 fast path | head access frequency만으로 quick 결정 (낮으면 바로 offload) |
| DRAM-affinity score의 hotness 반영 | decode step 기반 KV head reuse distance 반영 |
| CTC의 L2 way repurpose | prefetch 메타데이터(head access log)를 L2 side-way에 저장 |
| Traffic breakdown 분석 방법론 | demand/prefetch/fill/probe traffic 분해로 내 시스템 병목 진단 |

### 10-6. 충돌하거나 재검토 필요한 가정

1. **Granularity mismatch**: 이 논문은 4 KiB page + 256 B cacheline이고 내 연구는 수십 MB head. Head 단위 prefetch는 UVM page 단위와 완전히 다른 prefetch buffer 설계가 필요.
2. **HW-managed vs. SW-managed**: 이 논문의 DRAM cache는 HW-transparent이지만 내 연구의 head offloading은 attention kernel이 명시적으로 어느 head가 GPU/CPU에 있는지 알아야 함 → runtime scheduling과의 결합 필요.
3. **Latency scale**: SCM miss ~149 ns vs. PCIe access ~µs. 내 연구에서는 지연이 10× 이상 크므로 prefetch가 훨씬 더 중요하고, 잘못된 prefetch의 비용도 더 크다.

---

## 11. 핵심 용어 정리

| Term | 정의 (이 논문에서의 용법) |
|------|----------------------|
| HMS (Heterogeneous Memory Stack) | SCM die와 DRAM die를 TSV로 같은 channel에 적층한 3D stacked 메모리 |
| SCM (Storage-Class Memory) | DRAM보다 느리고 Flash보다 빠른 비휘발성 메모리; 이 논문에서는 PCM |
| PCM (Phase Change Memory) | 위상 변화 물질의 저항 차이를 이용한 SCM의 일종 (commercialized: Optane) |
| AMIL | Aggregated Metadata-In-Last-Column; DRAM cache row의 모든 tag를 마지막 column 32 B에 집약 |
| CTC (Configurable Tag Cache) | L2 cache의 일부 way를 tag cache로 전용, user-configurable 분할 |
| TAD (Tag-And-Data) | Alloy Cache 방식: 64 B column에 tag+data 동시 저장; ECC bit repurposing 필요 |
| SCM Penalty Score | `(Latency_SCM - Latency_DRAM) / NumColumnsAccessed`; per-access SCM overhead |
| DRAM-Affinity Score | SCM penalty score × page activation counter; bypass 결정의 최종 2D 기준 |
| RHBM | HBM 수용 용량 / workload footprint 비율; oversubscription 조절 손잡이 |
| Row buffer locality | row activation당 평균 column access 수; 높으면 SCM activation latency 분산 |
| tRCD | RAS to CAS Delay; row activation → column access 최소 지연 (SCM: 120 cyc vs. DRAM: 14 cyc) |
| tWR | Write Recovery; write 후 precharge/다음 activation 전 최소 지연 (SCM: 1000 cyc vs. DRAM: 16 cyc) |
| SLC/MLC/TLC | Single/Multi/Triple-Level Cell; bit/cell ↑ → 용량 ↑, tWR ↑, tRCD ↑ |
| TBN | Tree-Based Neighborhood prefetcher; NVIDIA UVM의 adaptive granularity prefetcher |
| Accel-sim | GPGPU-Sim 4.0 + SASS trace-driven frontend (ISCA'20, Khairy et al.) |
| UVMSmart | GPGPU-Sim 3.x에 UVM timing/functional simulation 추가 (Ganguly, ISCA'19) |
| Ramulator | 다양한 DRAM/메모리 표준 timing-accurate memory simulator (Kim et al., CAL'15) |
| AccelWattch | Accel-sim 기반 GPU power modeling framework (MICRO'21) |
| HotSpot | chip/package 열 모델링 툴 (Zhang et al., UVa) |
| FR-FCFS | First-Ready First-Come First-Served; row buffer 재사용 우선 DRAM scheduler |
| MSHR | Miss Status Holding Register; cache miss 추적용 entry |
| pdec | victim DRAM-affinity level 감소 확률 (`current_activation / max_activation`) |
| pLRU | pseudo-LRU; CTC에서 사용하는 4-bit approximation LRU |

---

## 12. 미해결 질문 / 더 읽을 것

### 논문이 남긴 Open Questions

1. **실제 SCM endurance**: DRAM cache write filtering이 PCM write endurance를 얼마나 연장하는가? Write-heavy workload에서 SCM 마모 시뮬레이션 없음.
2. **Full-scale validation**: 1/5 downscale 결과가 실제 A100(108 SM, 40 MiB L2)에서도 동일한가? L2 capacity 증가 시 CTC 효율은?
3. **Activation counter의 실제 이득**: disabled 상태로 평가했지만 ideal counter는 최대 7.6% speedup. 실제 counter 구현의 HW overhead vs. benefit 분석 없음.
4. **Multi-GPU HMS**: 논문은 single GPU 기준. Multi-GPU에서 HMS의 coherency와 inter-GPU traffic 관리는?
5. **HMS + NVLink 최적화**: HMS(NVLink)가 HBM(NVLink)보다 2.11× 빠르지만, NVLink cacheline-granularity cold access와 HMS DRAM cache 간의 추가 최적화 여지는?

### 더 읽으면 좋을 논문

| 논문 | 이유 | 링크 |
|------|------|------|
| Ganguly et al., "Interplay between HW prefetcher and page eviction policy in CPU-GPU UVM," ISCA'19 | 본 논문의 UVM model 원전, TBN prefetcher 상세 | [ACM DL](https://dl.acm.org/doi/10.1145/3307650.3322224) |
| Khairy et al., "Accel-Sim," ISCA'20 | Base simulator 원전 | [accel-sim.github.io](https://accel-sim.github.io/) |
| Chou et al., "BEAR," ISCA'15 | Primary baseline; CPU DRAM cache BW 최적화 기준 | [IEEE](https://ieeexplore.ieee.org/document/7284066) |
| Qureshi & Loh, "Alloy Cache (TAD)," MICRO'12 | AMIL이 극복하는 TAD 방식 원전 | — |
| Qureshi et al., "PCM as scalable DRAM alternative," ISCA'09 | PCM hybrid memory 연구의 기초 | [ACM DL](https://dl.acm.org/doi/10.1145/1555754.1555758) |
| Ustiugov et al., "Design guidelines for high-performance SCM hierarchies," MEMSYS'18 | SCM timing model의 출처 [131] | — |
| Li & Gao, "Baryon: Efficient hybrid memory management with compression," HPCA'23 | 최근 GPU hybrid memory 최적화, HMS와 orthogonal 결합 가능 | — |
| Kim et al., "Batch-aware unified memory in GPUs," ASPLOS'20 | Irregular workload UM 최적화 비교 연구 | — |
| Ganguly et al., "Adaptive page migration for irregular workloads under GPU oversubscription," IPDPS'20 | UVMSmart 확장, NVLink dynamic access counter 모델 원전 | [link](https://people.cs.pitt.edu/~debashis/papers/IPDPS2020.pdf) |
