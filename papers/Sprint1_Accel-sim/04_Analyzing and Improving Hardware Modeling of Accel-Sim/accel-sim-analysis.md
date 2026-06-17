# Analyzing and Improving Hardware Modeling of Accel-Sim

## 0. 한눈에 보기

- **출처**: CAMS 2023 (1st Workshop on Computer Architecture Modeling and Simulation), October 28, 2023, Toronto / Rodrigo Huerta, Mojtaba Abaie Shoushtary, Antonio González (UPC Barcelona) / DOI: https://doi.org/10.1145/3589236.3589244 / arXiv: 2401.10082
- **분류 태그**: #GPU-simulation #Accel-Sim #microarchitecture #sub-core #front-end #memory-pipeline #GPGPU
- **TL;DR**: Accel-Sim이 실제 NVIDIA sub-core 아키텍처를 세 군데(front-end, result bus, memory execution pipeline)에서 틀리게 모델링하고 있음을 지적하고, 더 현실적인 대안 모델을 제안해 일부 벤치마크에서 최대 23%의 사이클 수 변동을 확인한다.
- **핵심 기여**:
  1. Accel-Sim front-end를 sub-core별 private L0 instruction cache + 독립 fetch/decode unit 구조로 교체, instruction-to-cache-line 매핑 버그 수정
  2. Result bus에서 register file bank write-back 충돌 감지 로직 추가 (2-port 가정 반영)
  3. SM 전체 단일 memory pipeline을 sub-core별 분산 pipeline으로 교체하여 dispatch stall 구조 개선, address calculation과 coalescing을 별도 사이클로 분리
  4. 향후 과제로 WAR dependency handling, register file cache 모델링, NOC 계층(TPC/GPC), virtual memory, multi-GPU 식별
- **한 줄 직관**: Accel-Sim은 "sub-core 분할" 아키텍처를 선언했지만, front-end와 memory pipeline은 여전히 SM 전체를 단일 파이프라인으로 처리하는 근본적 모순이 있었으며, 이를 수정하면 시뮬레이션 결과가 벤치마크별로 크게 달라진다.

---

## 1. 문제 정의 & 동기

### 어떤 문제인가

(논문) Accel-Sim은 NVIDIA modern GPU를 targeting하는 cycle-accurate trace-driven simulator로, 연구 커뮤니티에서 널리 쓰인다. 그런데 현실의 NVIDIA GPU(Volta 이후)는 SM 내부를 **4개의 sub-core(processing block)**로 나누어 각각 독립적인 파이프라인을 돌린다. 문제는 Accel-Sim의 front-end와 memory execution pipeline이 이 sub-core 분할을 **제대로 반영하지 않고** 있다는 것이다.

### 왜 중요한가

시뮬레이터가 실제 HW와 다른 bottleneck 구조를 갖고 있으면, 그 위에서 내린 micro-architectural 결론이 실제 GPU에서는 달리 나타날 수 있다. 특히:
- Instruction cache hit/miss 패턴이 왜곡된다 → front-end bound 커널 분석이 부정확
- Memory pipeline의 dispatch stall 전파 방식이 달라진다 → memory-bound 커널의 IPC 예측이 틀릴 수 있다

### 기존 접근의 gap

| 구성요소 | Accel-Sim (기존) | 실제 HW | 문제 |
|---|---|---|---|
| Front-end fetch/decode | SM 전체 1개, 4 cycle/cycle 반복 | Sub-core마다 독립 fetch/decode + L0 I$ | 실제 L0 I$ miss 패턴 미반영, 서로 다른 커널의 instruction이 동일 주소로 충돌 |
| Instruction cache mapping | 비연속 주소 묶음, 커널 간 주소 공유 | PC 기반 cache line 매핑 | Hit을 miss로, miss를 hit으로 잘못 계산 |
| Memory pipeline | SM 전체 단일 dispatch latch, 단일 WB latch | Sub-core마다 memory unit | 한 sub-core의 memory stall이 전체 SM issue를 막음 |
| Coalescing | address calc + coalescing + request selection 동일 사이클 | (미공개, 추정 분리) | $C_{32,2} = 496$개의 address comparator 필요 → HW 비현실적 |
| Result bus | Register file bank 충돌 무시 | 2-port bank | 동일 bank에 동시 write-back 가능 여부 미검사 |

---

## 2. 핵심 아이디어

세 개의 fix는 모두 같은 방향을 가리킨다: **SM-level 단일 공유 자원을 sub-core-level 분산 자원으로 쪼개라.** Front-end는 L0 I$와 fetch/decode를 sub-core마다 독립으로 두고, memory pipeline은 sub-core마다 자체 memory unit을 두어 address latch와 WB latch를 분리한다. Result bus fix는 이와 직교하지만 역시 "bank conflict라는 실제 HW 제약"을 시뮬레이터에 반영하는 작업이다. 전체를 관통하는 통찰은 **"Accel-Sim은 throughput을 맞추기 위해 비용이 훨씬 큰 단일 공유 구조를 만들어뒀는데, 실제 HW는 비용을 낮추기 위해 분산 구조를 택했다"**는 것이다.

---

## 3. 배경 (이 논문 특유의 비자명한 전제)

**Sub-core warp 배분 공식**: (논문) 실제 NVIDIA GPU에서 `sub_core_id = warp_id % 4`. 즉 warp 0,4,8,…→ sub-core 0, warp 1,5,9,…→ sub-core 1 식으로 정적 배분. 이는 Zhe Jia et al.의 microbenchmark 논문[7][8]과 Barnes et al.[3]이 확인한 사실이다.

**Collector Unit (CU)**: Issue된 instruction이 모든 source operand가 RF에서 읽혀질 때까지 대기하는 버퍼. Sub-core당 CU 수는 유한하여(이 논문 설정: 2개/sub-core), CU가 꽉 차면 해당 sub-core에서 새 instruction issue가 막힌다.

**GTO (Greedy Then Oldest) scheduler**: 현재 실행 중인 warp를 계속 우선 선택하다가 stall되면 가장 오래된 eligible warp를 선택하는 issue policy.

**AVC (Absolute Variation in Cycles)**: 이 논문에서 정의한 메트릭. `|cycles_proposed - cycles_baseline| / cycles_baseline`. Speed-up과 달리 방향을 무시하므로, 일부 벤치마크는 느려지고 일부는 빨라지는 경우에도 평균 변화량을 파악하는 데 적합.

---

## 4. 제안 방법 상세

### 4.1 🎯 Front-end 개선

**문제의 구체적 메커니즘**: 기존 Accel-Sim은 1개의 fetch/decode unit이 4 cycle/GPU cycle로 모든 warp를 처리한다. 이 때문에 shared L0 I$가 매 cycle 4개의 access를 받아야 하고(실제 HW는 sub-core별 private L0이므로 각 1개), instruction을 cache line에 매핑할 때 PC 연속성을 고려하지 않아 다음 두 버그가 발생한다:

1. **Branch 후 비연속 PC packing**: PC 0x100의 instruction과 PC 0x460의 instruction이 같은 fetch packet으로 묶여, 서로 다른 cache line에 속하는 instruction이 한 번에 hit처럼 처리된다.
2. **커널 간 주소 충돌**: 서로 다른 kernel의 첫 두 instruction이 동일 메모리 주소를 가진다고 가정하여, 실제로는 miss여야 할 access가 hit으로 계산된다.

**제안 모델**:

```
Per sub-core (×4):
  - Private L0 instruction cache (16 KB)
  - RR fetch unit: Warps {0,4,8,…,28}, {1,5,9,…,29}, {2,6,10,…,30}, {3,7,11,…,31}
  - Decode unit
  - Sub-core pipeline

SM-level shared:
  - L1 instruction cache (32 KB)
  - Round-robin priority arbiter (4 sub-core → L1 I$ 요청 중재)
```

Fix 내용:
- PC를 올바르게 cache line에 매핑 (연속된 주소만 같은 line에 묶음)
- 커널마다 다른 가상 주소 공간 할당 (커널 간 주소 충돌 제거)
- L0 I$ max requests/replies = 1 (sub-core별 독립 접근, Table 1)

### 4.2 Result Bus 개선

**문제**: 기존 simulator는 fixed-latency instruction dispatch 시, 해당 latency 이후에 result bus가 비어 있는지만 확인한다. Register file bank write-back 충돌은 체크하지 않는다.

**제안**: destination register의 RF bank를 추적하여, 동일 사이클에 동일 bank에 write-back하려는 instruction 수가 bank port 수를 초과하면 dispatch를 막는다.

- (논문) NVIDIA 최신 아키텍처는 RF bank당 2개 port를 가지며(Jia et al. 확인), 각 port는 write-back 또는 read로 사용 가능
- 따라서 동일 destination RF bank로 향하는 instruction 최대 2개가 동일 사이클에 write-back 완료되도록 schedule 가능

### 4.3 🎯 Memory Execution Pipeline 개선

**기존 모델의 구조적 문제 분해**:

| 문제 | 세부 내용 | 영향 |
|---|---|---|
| 단일 dispatch latch | SM 전체 1개. instruction이 모든 request를 보낼 때까지 latch에 묶임 | 한 instruction의 coalescing stall이 다른 sub-core issue까지 중단 |
| CU 고갈 | dispatch latch에 묶인 memory instruction이 CU를 점유 → 새 issue 불가 | Warp-level 병렬성 저하 |
| 동일 사이클 address calc + coalescing | 32-thread warp, 모든 쌍 비교: $C_{32,2} = \binom{32}{2} = 496$ address comparators | HW 비현실적 |
| 순서 강제 request 전달 | 생성 순서대로만 memory structure에 전달 | 빈 bank/memory structure가 있어도 활용 불가 |
| 단일 WB latch | SM 전체 공유 → 서로 다른 sub-core의 ready access 간 경합 | WB latch 경합 시 이전 instruction이 block하면 후속도 지연 |

**제안 모델 (per sub-core memory unit)**:

```
[Dispatch latch] → 1 cycle (address calc 시작, 비어있으면 바로 빠져나감)
       ↓
[Address latch] → N cycles (thread 당 1 cycle씩 순차 address 계산, comparators = 32)
       ↓
[Request buffer] → 생성된 request 저장
       ↓
[Round-robin arbiter between sub-cores] → L1D bank / shared memory / constant / texture 중 유휴 구조 우선 선택
       ↓
[Per sub-core WB latch] + [WB arbiter between sub-cores]
```

핵심 설계 선택:
- Address calculation과 coalescing을 **사이클 분리**: instruction은 dispatch latch에 최대 1 cycle 체류 (address latch가 비어있는 경우)
- Request는 **생성 순서 무관** 전달: arbiter가 빈 memory structure 발견 시 해당 request를 먼저 처리
- Per sub-core WB latch → sub-core 간 WB 경합을 arbiter로 처리

(논문) 단, 실제 상용 GPU의 memory unit 상세 구조는 공개되어 있지 않으므로, 이 설계는 "reasonable aggressive performance design"이며 실제 HW와 다를 수 있음을 저자가 명시.

---

## 5. 시스템 · HW 측면

### Simulation 환경

(논문) Accel-Sim을 **trace-driven** 모드로 사용. Cycle-accurate kernel execution simulation이지만, trace는 실제 GPU에서 미리 수집된 instruction trace를 재생하는 방식 (execution-driven이 아님).

Target HW: **NVIDIA RTX 2070 Super** (Turing 아키텍처, TU104)

### GPU 설정 (Table 1 원문 그대로)

| Parameter | Value |
|---|---|
| Clock | 1605 MHz |
| SP/INT/SFU/Tensor Units per sub-core | 1/1/1/1 |
| Warps per SM | 32 |
| Warp Width | 32 |
| Number of registers per SM | 65536 |
| Issue Scheduler policy | GTO |
| Number of SMs | 40 |
| Sub-cores per SM | 4 |
| Number of Collector Units per sub-core | 2 |
| L1 instruction cache size | 32 KB |
| L1 data cache size | 32 KB |
| Shared memory size | 64 KB |
| L2 cache size | 4 MB |
| Memory Partitions | 16 |
| **[Fix] L0 instruction cache size** | **16 KB** |
| **[Fix] Max. Num. requests and replies of L0I** | **1** |
| **[Fix] Register file ports per bank** | **2** |

### HW vs SW 구현 분담

(논문) Accel-Sim은 순수 SW simulator이므로 RTL/합성 없음. 제안된 모델은 모두 simulator의 C++ 코드 수정으로 구현. RTL 검증은 없으며, 상용 GPU 동작과의 직접 비교(validation against real HW cycle count)도 이 논문에서는 수행하지 않음.

---

## 6. 🎯⭐ 실험 셋업 & 재현 관점

### Benchmark Suite

(논문) 총 42개 벤치마크, 완전 실행(simulation to completion):

| Suite | 특성 |
|---|---|
| Rodinia 3.1 | HPC 다양한 병렬 패턴 |
| DeepBench (Baidu) | DNN primitive (GEMM, conv, RNN) |
| Parboil | Scientific computing |
| Pannotia | Irregular graph applications |
| ISPASS-2009 | 클래식 GPGPU 벤치마크 |

결과에서 significant한 8개 벤치마크만 Figure 6에 표시: `fw`, `dwt2d`, `kmeans`, `lud`, `conv-train`, `gemm-train`, `rnn-lstm-train`, `rnn-gru-train`

### Baseline

기존 Accel-Sim (수정 전) 대비 비교. 각 fix를 개별(Front-end only / Result bus only / Memory pipeline only)과 통합(All) 네 가지 configuration으로 측정.

### Metrics

- **Speed-up**: `(proposed_cycles / baseline_cycles)`. 양의 값이면 빨라진 것.
- **AVC (Absolute Variation in Cycles)**: `|proposed_cycles - baseline_cycles| / baseline_cycles`. 방향 무시. 이 논문의 primary metric.

### 🎯 재현 체크리스트

| 항목 | 상태 |
|---|---|
| Simulator codebase | Accel-Sim GitHub (공개), 단 이 논문의 수정 코드는 별도 공개 여부 **논문에 명시 없음** |
| Benchmark 입력 및 실행 파라미터 | **명시 없음** (Rodinia/DeepBench 등은 공개이나 구체적 실행 옵션 미보고) |
| Trace 파일 | **제공 없음** (trace-driven이므로 동일 trace 없이는 사이클 정확 재현 불가) |
| GPU configuration (.config 파일) | Table 1으로 재구성 가능하나 Accel-Sim config 파일 자체 미첨부 |
| 42개 중 Figure 6에 표시된 8개 외 나머지 결과 | **미보고** |
| 실제 RTX 2070 Super 측정값 대비 validation | **없음** |

(분석) 이 논문은 "실제 GPU 대비 정확도 향상"을 주장하는 것이 아니라 "더 현실적인 모델 구조"를 제안하는 것이며, HW validation 부재는 워크샵 논문의 한계로 받아들여야 한다.

---

## 7. 결과 분석

### 전체 평균

(논문) 42개 벤치마크 평균:
- Speed-up: **+0.25%** (거의 변화 없음)
- AVC: **3.67%** (평균적으로 약 3.67%의 사이클 수 변화)

Speed-up < AVC인 이유: 일부 벤치마크는 빨라지고 일부는 느려지므로 방향이 상쇄됨.

### 개별 벤치마크 결과 (Figure 6, 원문 수치)

| Benchmark | Front-end AVC | Result bus AVC | Memory pipeline AVC | All AVC |
|---|---|---|---|---|
| fw | ~0% | ~0% | **~23%** | **~21%** |
| dwt2d | **~17%** | ~0% | ~2% | ~18% |
| kmeans | ~2% | ~0% | ~3% | ~3% |
| lud | **~12.36%** | ~0% | ~3% | ~13% |
| conv-train | ~1% | ~1% | ~5% | ~6% |
| gemm-train | ~3% | ~1% | ~5% | **~12%** |
| rnn-lstm-train | ~2% | ~0% | ~4% | ~5% |
| rnn-gru-train | ~2% | ~0% | ~3% | ~4% |

※ Figure 6에서 시각적으로 읽은 근사치. 논문에 정확한 수치 표 없음.

### 이득이 어디서 오는가

**Memory pipeline fix** → `fw`에서 23% AVC. `fw`(Floyd-Warshall)는 irregular memory access 패턴으로 coalescing 효율이 낮아 dispatch latch 체류 시간이 길고, 단일 dispatch latch 모델에서 cross-sub-core stall 전파가 심해지는 케이스다.

**Front-end fix** → `dwt2d`(17%), `lud`(12.36%). Figure 7에서 이 두 벤치마크의 L1 instruction cache miss rate increment factor가 ~2x, ~4x로 크게 증가하는 것을 확인. 즉, 기존 모델은 이 워크로드에서 I$ miss를 과소평가하고 있었으며, 수정 후 miss가 더 많이 발생해 사이클 수가 증가(느려짐).

**Result bus fix** → 모든 벤치마크에서 영향이 가장 작음. Bank conflict 빈도 자체가 낮기 때문.

**gemm-train에서 All=12%**: front-end(~3%)와 memory(~5%)가 **같은 방향**으로 작용하여 합산 효과 발생. 다른 벤치마크에서는 두 fix가 반대 방향으로 작용해 상쇄되는 경우도 있음.

**Miss rate increment factor의 역설** (논문): `gemm-train`의 L1 I$ miss rate increment factor가 가장 크지만(Figure 7: ~10x), AVC 영향은 `dwt2d`/`lud`보다 작다. 이유는 `gemm-train`의 총 실행 사이클이 `dwt2d`보다 27.4x, `lud`보다 7.8x 길어서, instruction cache miss 추가 cost의 비율이 상대적으로 작아지기 때문.

---

## 8. 비판적 분석

### 강점

(분석) 세 가지 fix 모두 실제 HW 아키텍처 문서(NVIDIA whitepaper, Zhe Jia microbenchmark)에 근거한 well-motivated 수정이다. 특히 front-end의 cache line 매핑 버그는 trace-driven simulator 특유의 artifact로, 이를 명확히 지적한 것은 재현성 관점에서 의미있는 기여다.

### 한계

1. **HW validation 없음**: (분석) "더 현실적"이라고 주장하지만, 실제 RTX 2070 Super에서 측정한 사이클 수와의 비교가 없다. Accel-Sim 원 논문[9]은 IPC 기준 일정 수준의 validation을 제시했는데, 이 수정이 validation accuracy를 높이는지는 미확인. 워크샵 논문의 한계로 이해되지만, 연구에 사용 전 주의 필요.
2. **Memory pipeline이 "aggressive"임을 인정**: (논문) 실제 상용 GPU memory unit 구조가 미공개이므로 저자가 직접 "aggressive performance design that might differ from commercial designs"라고 명시. 제안 모델이 오히려 실제보다 낙관적일 가능성.
3. **42개 중 8개만 보고**: (분석) 전체 결과를 표로 제공하지 않아, 나머지 34개 벤치마크에서 어떤 경향이 있는지 알 수 없다. "significant"한 케이스만 선택적 제시는 평균 3.67% AVC가 어떻게 분포하는지 파악을 어렵게 한다.
4. **sub-core가 4개가 아닌 GPU 미고려**: (논문) 현재 설정은 4 sub-core 고정. 다른 수의 sub-core를 갖는 GPU로 일반화 가능성 미검토.
5. **코드 미공개 확인 필요**: 논문에 코드 공개 링크가 없다. 재현 불가 위험.

### 숨은 가정

- (분석) Sub-core memory pipeline의 "address 계산: thread당 1 cycle" 가정은 arbitrary. 실제 SIMD 실행에서 32-thread 전체 address 계산이 1 cycle에 완료될 수도 있다.
- (분석) CU 수를 2개/sub-core로 설정했는데, 실제 Turing의 CU 구성은 미공개이며 이 파라미터가 결과에 상당한 영향을 줄 수 있다.

---

## 9. 관련 연구 맥락

### Accel-Sim 원 논문

(외부) Khairy et al., "Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling," ISCA 2020 [9]. GPGPU-Sim을 기반으로 trace-driven 모드를 추가하고, 다수 NVIDIA 아키텍처에 대한 validation을 제시한 GPU simulator 논문. 이 논문의 직접적인 수정 대상.

### Microbenchmark 레퍼런스

(외부) Zhe Jia et al., "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking," arXiv 2018 [8]; "Dissecting the NVidia Turing T4 GPU via Microbenchmarking," 2019 [7]. Sub-core 구조, register file port 수 등 이 논문의 HW 근거를 제공하는 reverse-engineering 연구.

### Barnes et al. HPCA 2023

(외부) Barnes, Shen, Rogers, "Mitigating GPU Core Partitioning Performance Effects," HPCA 2023 [3]. Sub-core 분할로 인한 성능 불균형(특정 warp 집합이 특정 sub-core에 고정됨으로써 발생하는 load imbalance)을 분석하고 완화하는 연구. 이 논문과 같은 warp-to-sub-core 배분 공식(`warp_id % 4`)을 확인.

### NVArchSim

(외부) Villa et al., "Need for Speed: Experiences Building a Trustworthy System-Level GPU Simulator," HPCA 2021 [21]. NVIDIA 내부 simulator 구축 경험. 이 논문은 학술 simulator(Accel-Sim)와의 gap을 메우려는 시도로 이해할 수 있다.

### Novelty의 위치

(분석) 이 논문의 기여는 새로운 micro-architectural idea 제안이 아니라 **기존 simulator의 구조적 모델링 버그를 진단하고 수정**하는 데 있다. CAMS 워크샵 (simulator/modeling 전문 워크샵) 성격에 맞는 contribution으로, simulator user의 결과 신뢰도에 실질적 영향을 미치는 작업이다.

---

## 10. 🎯⭐ 내 연구와의 연결점 · 적용 가능성

### Accel-Sim 기반 연구에서 결과 신뢰도 방어 논거

(분석) 내 연구가 Accel-Sim 기반 simulation을 사용한다면, 이 논문을 다음과 같이 활용할 수 있다:

1. **"비현실적 모델링" 비판 방어**: Accel-Sim의 memory pipeline과 front-end 모델이 실제 HW와 다르다는 것은 이미 이 논문이 지적했다. 내 연구 결과를 보고할 때, "제안 기법이 Accel-Sim의 현재 모델링 한계 내에서 X% 이득을 보인다"고 framing하거나, 반대로 이 논문의 fix를 적용한 버전 위에서 실험하면 더 defensible해진다.

2. **Memory-bound 워크로드에서 특히 주의**: `fw`에서 memory pipeline fix만으로 23% AVC가 발생했다. LLM inference는 KV cache access로 인해 memory-bound 구간이 많으므로, **unmodified Accel-Sim으로 측정한 memory-bound 커널의 사이클 수치는 ±20% 수준의 불확실성을 가정해야 한다.** 이를 논문 limitation으로 명시하거나, fix 적용 버전으로 재실험하는 것을 권장.

3. **Instruction cache 민감 워크로드**: MoE/MoH inference에서 expert switch가 빈번하면 각 expert의 instruction footprint가 다르고, 이 논문이 지적한 "커널 간 주소 충돌" 버그가 특히 심각하게 나타날 수 있다. Expert dispatch 패턴이 I$ miss rate에 미치는 영향이 unmodified Accel-Sim에서는 과소평가될 가능성이 있다.

### Prefetch 엔진 hook 지점 후보 식별 🎯

이 논문의 수정 모델을 기준으로, prefetch 로직을 끼워 넣을 수 있는 구조적 hook 지점은 다음과 같다:

| Hook 위치 | 설명 | 적합한 prefetch 유형 |
|---|---|---|
| **Sub-core Memory Unit → Request Buffer** | Coalescing 완료 후 request가 buffer에 쌓이는 시점. 여기서 miss 여부를 판단해 L2/DRAM prefetch 트리거 가능 | L1D miss-based prefetch |
| **Round-robin arbiter between sub-cores** | Sub-core 간 memory request를 중재하는 지점. 특정 sub-core의 access pattern을 관찰해 stride prefetch 가능 | Stride/stream prefetch |
| **L1 instruction cache arbiter (L1 I$ → sub-core L0)** | L0 miss로 L1에 접근하는 시점. Expert code prefetch 트리거 가능 | Instruction prefetch (MoE expert 전환 예측) |
| **Per sub-core WB latch** | 응답이 돌아오는 시점. Miss latency 측정으로 adaptive prefetch distance 조정 가능 | Adaptive prefetch |

(분석) 가장 자연스러운 위치는 **Sub-core Memory Unit 내 Request Buffer 이후 (L1D miss detect 시점)**이다. 기존 단일 memory pipeline 모델에서는 이 지점이 SM 전체 공유이므로 sub-core 수준의 access pattern 추적이 어려웠지만, 제안 모델에서는 per-sub-core access pattern을 독립적으로 추적할 수 있어 sub-core 수준 prefetch policy가 가능해진다.

### Offloading 연구와의 연결

(분석) MoE/MoH expert offloading 연구에서 expert weight를 DRAM/host memory에서 SM으로 가져오는 것이 핵심 bottleneck이다. Accel-Sim의 memory partition 모델(16개 partition, L2 4MB)은 이 과정을 시뮬레이션하는 기반이다. 이 논문은 SM 내부 병목(memory pipeline)이 어떻게 작동하는지를 수정했으므로, offloading scenario에서 **"SM이 prefetch된 데이터를 얼마나 빠르게 소화하는가"** 의 시뮬레이션 정확도에 직접 영향을 준다. Prefetch로 데이터는 L2까지 왔지만, sub-core memory pipeline의 dispatch latch stall이 실제 소비 속도를 제한할 수 있다 — 이 bottleneck이 unmodified Accel-Sim에서는 과장되어 있을 수 있다(단일 dispatch latch가 현실보다 더 많이 stall).

### INT4 quantization과의 연관

(분석) INT4 weight를 사용하면 per-access data size는 줄어들지만 thread당 dequantization 연산이 추가된다. 이는 execution unit 측 bottleneck이며, memory pipeline fix와는 orthogonal. 단, coalescing 패턴이 바뀔 수 있으므로(INT4 packed load의 byte 정렬) 제안된 memory pipeline 모델에서의 address calculation 사이클 수 가정을 재검토할 필요가 있다.

---

## 11. 핵심 용어 정리

| Term | 이 논문에서의 정의 |
|---|---|
| Sub-core (processing block) | SM 내부의 독립 pipeline 단위. NVIDIA Volta 이후 SM당 4개. 각자 fetch/decode/issue/operand collection/execution 단계를 독립 운영 |
| L0 instruction cache | Sub-core 전용 private instruction cache. L1 I$의 하위 계층 |
| Collector Unit (CU) | Issue된 instruction이 모든 source operand RF 읽기를 기다리는 버퍼. Sub-core당 유한 개수(이 논문: 2개) |
| AVC (Absolute Variation in Cycles) | `|proposed - baseline| / baseline`. 방향 무시한 cycle 변화량 |
| Dispatch latch | Memory instruction이 address 계산 및 request 전송 중 머무는 latch. 기존: SM 전체 공유 1개 → 제안: sub-core별 1개 |
| Address latch | 제안 모델에서 새로 도입. Thread-by-thread address 계산 중 instruction이 체류 |
| Request buffer | Coalescing 완료된 memory request 저장 버퍼. 제안 모델에서 out-of-order request 전달 가능 |
| GTO (Greedy Then Oldest) | Issue scheduler policy. 현재 warp 우선, stall 시 oldest eligible warp |
| WB latch (write-back latch) | Memory 응답이 도착해 RF에 쓰기 전 대기하는 latch |
| Miss rate increment factor | 수정 모델에서의 miss rate / 기존 모델에서의 miss rate 비율 (Figure 7) |

---

## 12. 미해결 질문 / 더 읽을 것

### 논문이 남긴 open questions

1. **WAR dependency handling**: Mishkin [10]이 지적한 out-of-order dispatch에서의 WAR hazard를 어떻게 수정할 것인가. Scoreboard 없는 modern GPU의 dependency 감지 메커니즘이 미공개.
2. **Register file cache**: NVIDIA Malekeh [1] 같은 RF cache 제안이 실제 industry 대비 얼마나 개선인지, RF cache가 없는 Accel-Sim으로는 판단 불가.
3. **NOC hierarchy (TPC/GPC)**: SM ↔ memory partition 간 flat NOC가 실제 hierarchical interconnect와 얼마나 다른지, 특히 Hopper의 distributed shared memory 기능 지원에 필요.
4. **Virtual memory & multi-GPU**: 현재 Accel-Sim이 미지원. LLM inference의 tensor parallelism 시뮬레이션에 필요.
5. **제안 memory pipeline이 실제 HW와 맞는가**: Validation 없이 "reasonable"이라고만 주장. 실제 RTX 2070 Super 사이클과 비교 필요.

### 더 읽을 논문

- **(외부)** Khairy et al., "Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling," ISCA 2020. https://doi.org/10.1109/ISCA45697.2020.00047 — 이 논문의 직접 수정 대상. Accel-Sim validation 방법론 이해 필수.
- **(외부)** Barnes, Shen, Rogers, "Mitigating GPU Core Partitioning Performance Effects," HPCA 2023. https://doi.org/10.1109/HPCA56546.2023.10070957 — Sub-core 분할의 성능 불균형 문제와 mitigation. 이 논문의 front-end fix와 직접 연관.
- **(외부)** Zhe Jia et al., "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking," arXiv 2018. https://arxiv.org/abs/1804.06826 — Sub-core 구조, RF port 수 등 HW 근거 원전.
- **(외부)** Villa et al., "Need for Speed: Experiences Building a Trustworthy System-Level GPU Simulator," HPCA 2021. https://doi.org/10.1109/HPCA51647.2021.00077 — NVIDIA 내부 simulator 관점. Accel-Sim의 accuracy gap을 이해하는 데 참고.
- **(외부)** Abaie Shoushtary et al., "Lightweight Register File Caching in Collector Units for GPUs," GPGPU 2023. https://doi.org/10.1145/3589236.3589245 — 이 논문 저자 중 한 명(Abaie)의 동시기 work. RF cache 모델링 공백이 simulation에 미치는 영향과 연관.
