# A Survey on Inference Optimization Techniques for Mixture of Experts Models

## 0. 한눈에 보기

- **출처**: ACM Computing Surveys, Vol. 58, No. 10, Article 247, March 2026 / DOI: [10.1145/3794845](https://doi.org/10.1145/3794845)
- **저자**: Jiacheng Liu\*, Peng Tang\* (공동 1저자), Wenfeng Wang, Yuhang Ren, Xiaofeng Hou, Pheng Ann Heng, Minyi Guo, Chao Li — 상하이교통대·홍콩중문대 공동
- **분류 태그**: `#MoE` `#LLM-inference` `#expert-offloading` `#prefetching` `#quantization` `#expert-parallelism` `#HW-SW-codesign` `#survey`
- **TL;DR**: MoE 모델 inference 최적화를 model / system / hardware 3-level taxonomy로 체계화한 최초의 전용 survey로, 특히 memory-constrained 환경에서의 expert offloading·prefetching 기법들을 가장 촘촘하게 커버한다.
- **핵심 기여**:
  1. MoE inference에 특화된 3-level taxonomy(model-level / system-level / hardware-level) 최초 제시
  2. 30개 이상의 SoTA MoE 모델 파라미터 스펙을 단일 표로 정리 (Table 1)
  3. Expert pruning / quantization / distillation 비교 표 제공 (Table 2, 3, 4)
  4. Expert offloading 시스템들의 성능을 동일 baseline(Mixtral-8x7B)로 정규화해 비교 (Table 8, 9)
  5. 오픈 리포지토리([github.com/MoE-Inf/awesome-moe-inference](https://github.com/MoE-Inf/awesome-moe-inference/)) 운영으로 지속 업데이트
- **한 줄 직관**: MoE의 sparse activation은 "필요한 것만 불러온다"는 장점이지만, 동시에 "무엇이 필요할지 모른다"는 예측 문제를 만든다 — 이 survey는 그 예측 문제를 푸는 모든 방식의 지도다.

---

## 1. 문제 정의 & 동기

### 왜 MoE inference가 어려운가

Dense LLM과 달리 MoE는 **conditional computation** 구조다. 각 토큰마다 router가 $N$개의 expert 중 top-$K$를 선택하여 실행한다. 이는 다음 세 가지 독립적인 어려움을 만든다:

| 난제 | 원인 | 구체적 증상 |
|---|---|---|
| **메모리 압박** | 전체 expert weight는 모두 어딘가에 있어야 함 | Mixtral-8x7B(46.7B)은 FP16 기준 ~93GB; A100 1장 불가 |
| **동적·불균형 연산** | 어떤 expert가 선택될지 입력에 따라 달라짐 | 일부 expert에 토큰 집중 → 특정 GPU idle |
| **Expert 로딩 지연** | offloading 시 cache miss 발생 | HOBBIT 측정: expert loading이 전체 inference 시간의 >80% |

### 기존 접근의 gap

(논문) 기존 LLM efficiency survey [10, 92, 204, 239]와 MoE architecture survey [14, 45, 188]는 존재하지만, **MoE inference optimization에 특화된** 체계적 taxonomy는 없었다. 이 논문이 최초.

---

## 2. 핵심 아이디어

이 survey의 organizing insight는 하나다: MoE inference의 모든 최적화는 결국 **"어떤 expert가, 언제, 어디서, 어떤 정밀도로 필요한가"를 더 잘 알거나(예측), 더 싸게 처리하거나(압축/병렬화), 더 빠르게 공급하는(memory hierarchy, HW) 세 범주 중 하나**다. 이를 model / system / hardware 3-level로 decompose하면 각 레이어의 최적화 목표가 명확해진다.

---

## 3. 배경 (비자명한 전제만)

### MoE forward pass의 정확한 수식 흐름

$$\theta = \text{Softmax}(R(x)), \quad x \in \mathbb{R}^d \tag{1}$$

$$E_{\text{selected}} = \text{TopK}(\theta, K) \tag{2}$$

$$y_i = E_i(x), \quad \forall i \in E_{\text{selected}} \tag{3}$$

$$y = \sum_{i \in E_{\text{selected}}} \frac{\theta_i}{\sum_{j \in E_{\text{selected}}} \theta_j} \cdot y_i \tag{4}$$

(논문) Switch Transformer는 normalization 없이 $y = \sum \theta_i y_i$ (식 5)를 사용. Qwen2-MoE는 `norm_topk_prob` 파라미터로 선택 가능하게 구현. 이처럼 normalization 여부는 **설계 선택지**이며 구현마다 다르다.

### Expert 구성의 다양성: Table 1에서 읽을 것

(논문) Expert 표기는 `활성화수/전체수/공유expert수` 형식. 예를 들어 DeepSeek-V3는 `8/256/1`로 256개 중 8개를 활성화하고 1개는 항상 공유. 공유 expert(shared expert)는 모든 토큰에 항상 적용되는 dense 성분으로, DeepSeekMoE 이후 주류화됨.

---

## 4. 제안 방법 상세 — Taxonomy 구조

```
MoE Inference Optimization
├── Model Level
│   ├── Architecture Design
│   │   ├── MoE-based Attention (MoH, JetMoE, MoA, SwitchHead, BAM, MAE, ...)
│   │   └── MoE-based FFN (MoE++, Pre-gated MoE, SCoMoE, COMET, MoELoRA, ...)
│   ├── Model Compression
│   │   ├── Expert Pruning    (structured delete / merge)
│   │   ├── Expert Quantization (1~8bit)
│   │   ├── Expert Distillation (sparse→dense, sparse→sparse)
│   │   └── Expert Decomposition (low-rank / tensor decomposition)
│   └── Algorithm Improvement
│       ├── Dynamic Gating    (adaptive top-k)
│       └── Sparse to Dense   (dense 모델로 변환)
├── System Level
│   ├── Expert Parallelism
│   │   ├── Parallel Strategy (Tutel, Alpa, SmartMoE, ...)
│   │   ├── Load Balancing    (Prophet, Lazarus, FlexMoE, ...)
│   │   ├── All-to-all Comm.  (Janus, ExFlow, Aurora, LocMoE, ...)
│   │   └── Task Scheduling   (ScMoE, ScheMoE, PipeMoE, ...)
│   └── Expert Offloading
│       ├── Prefetching       (adjacent-layer prediction, learned predictor)
│       ├── Caching           (LRU, LFU, LHU, static profiling)
│       ├── Expert Loading    (mixed-precision loading)
│       └── CPU Assisting     (Fiddler, HOBBIT, MoE-Lightning)
└── Hardware Level
    ├── NDP/PIM-based         (MoNDE, Duplex)
    ├── FPGA                  (FLAME, M3ViT, Edge-MoE)
    └── Custom Accelerator    (Space-Mate)
```

---

## 5. 시스템 · HW 측면

### Expert Parallelism — All-to-all 병목

(논문) Expert parallelism의 핵심 구조: 각 GPU는 전체 expert의 부분집합만 보유하고 non-expert 파라미터(attention, embedding)는 모두 복제. MoE layer 실행 시:

1. 각 GPU에서 attention + router 계산
2. **All-to-all #1**: 토큰을 담당 expert가 있는 GPU로 재분배
3. 각 GPU에서 expert 계산
4. **All-to-all #2**: 결과를 원래 GPU로 복귀

이 two-pass all-to-all이 cloud 환경의 핵심 bottleneck. 주요 최적화 방향:

| 접근 | 대표 논문 | 핵심 아이디어 |
|---|---|---|
| Hierarchical A2A | Tutel, HetuMoE, DeepSpeed-MoE | intra-node 먼저, inter-node 나중 |
| Token 대신 Expert 이동 | Janus | 데이터보다 파라미터를 옮기는 게 더 쌈 |
| A2A 횟수 감소 | ExFlow | inter-layer expert affinity 활용해 2→1회 |
| A2A 순서 최적화 | Aurora | bandwidth contention 방지하는 전송 순서 도출 |

### Expert Offloading — Memory Hierarchy

(논문) Figure 7(b)의 3-tier 구조:

```
GPU HBM (<80GB)   : Non-expert weights (attention, norms) + Expert Cache (hot experts)
CPU DDR (<1TB)    : 나머지 expert weights
SSD (>1TB)        : 전체 모델 (필요시)
```

GPU→CPU 대역폭 ~32GB/s, CPU→SSD ~3GB/s. Expert loading이 전체의 >80%를 차지하는 이유가 여기에 있다.

---

## 6. 🎯⭐ 실험 셋업 & 재현 관점

### 평가 모델 (Benchmarking Workloads)

(논문) 대부분의 offloading 시스템은 두 모델에 집중:

| 모델 | 스펙 | 사용 이유 |
|---|---|---|
| **Mixtral-8x7B** | 46.7B 파라미터, 2/8 expert | 가장 널리 쓰이는 오픈소스 MoE |
| **Switch Transformer** | Google, 다양한 크기 | 초기 MoE 연구의 표준 baseline |

(논문) 논문이 명시적으로 권고: Switch Transformer는 시스템마다 baseline이 달라 비교 어렵고, **Mixtral-8x7B이 더 적합한 평가 workload**라고 결론.

### Expert Parallelism 성능 비교 (Table 5, 6)

**Training speedup (vs DeepSpeed-MoE)**

| 시스템 | Speedup | 지표 |
|---|---|---|
| HetuMoE | 5.88× | Time per iteration |
| Lazarus | 3.40× | Tokens per second |
| Parm | 3.00× | Time per iteration |
| Prophet | 2.39× | Time per iteration |
| FlexMoE | 1.70× | Time to converge |
| DeepSpeed-TED | 1.35× | Time per iteration |
| TA-MoE | 1.31× | Tokens per second |

**Inference speedup**

| 시스템 | Speedup | Baseline | 핵심 기법 |
|---|---|---|---|
| Brainstorm | 3.33× | Tutel | 프로파일링 기반 weight preload |
| MoE-Deploy | 3.32× | Tutel | 고빈도+저빈도 expert 결합 |
| ExFlow | 2.2× | DeepSpeed-MoE | A2A 2→1 감소 |
| Tutel | 2.03× | Fairseq | 동일 layout으로 전략 전환 무비용화 |
| Aurora | 1.81× | Lina | 전송 순서 최적화 |
| Lina | 1.63× | DeepSpeed | A2A에 allreduce 우선순위 양보 |

### Expert Offloading 성능 비교 (Table 8 — Mixtral-8x7B 기준)

| 시스템 | Speedup | Baseline | 핵심 기법 |
|---|---|---|---|
| **Fiddler** | **8.20×** | Mixtral-Offloading | CPU에서 expert 연산 실행 |
| MoE-Lightning | 3.50× | FlexGen | CPU-GPU-I/O pipeline |
| HOBBIT | 2.30× | MoE-Infinity | 적응형 정밀도 expert 로딩 |
| Mixtral-Offloading | 2.28× | Accelerate | LRU 캐시 |
| MoE-Infinity | 1.96× | Mixtral-Offloading | request-level 빈도 추적 prefetch |
| ExpertFlow | 1.99× | CacheMoE | 학습된 predictor로 prefetch |
| AdapMoE | 1.36× | Mixtral-Offloading | 불필요 expert skip + prefetch |
| ProMoE | 1.16× | LRU-MoE | sliding-window MLP predictor |

### Expert Quantization 비교 (Table 3)

| 방법 | 메모리 감소 | 정확도 손실 | 추론 속도 향상 | Bit |
|---|---|---|---|---|
| MC-MoE | 4.27× | 3.8% | 1.80× | 1, 2, 3 |
| MoQE | 4.90× | 0.97% | — | 2, 3, 4 |
| MoE-CSP | 4.00× | — | 26.00× | 4, 8 |
| HOBBIT | — | 1% | 1.35× | 2, 4 |
| EdgeMoE | 1.05~1.18× | 5% | 1.11~2.78× | 2, 4, 8 |
| QMoE | 20× | 6.7% | **0.95×** (overhead!) | 1, 2 |
| CMoE | 150× | 23.81% | — | 1, 2, 4 |

(분석) QMoE는 20× 메모리 감소에도 불구하고 추론 속도가 오히려 5% 느려짐 — 전용 1-bit CUDA kernel 부재 때문. 하드웨어 지원 없는 극단적 quantization의 함정을 잘 보여준다.

### Dynamic Gating 비교 (Table 4)

| 방법 | FLOPs 감소 | Speedup | 임계값 전략 |
|---|---|---|---|
| Fixed top-k | 0% | 1.0× | — |
| Li et al. | 38.2% | 1.32× | 누적 확률 |
| DynMoE | 9% | 1.37× | 단일 expert 확률 |
| XMoE | 75% | — | 누적 확률 |
| AdapMoE | 25% (expert 수 기준) | 1.35× | 민감도 기반 |

### 재현 체크리스트

**명시된 정보:**
- 평가 모델: Mixtral-8x7B, Switch Transformer (대부분의 시스템)
- 주요 framework: PyTorch (parallelism 12편), Transformers (offloading 7편)
- 측정 지표: tokens/second, latency/token, time/iteration (시스템마다 상이)

**논문에 명시 없음 / 재현 장벽:**
- 각 시스템의 GPU 메모리 크기, GPU 모델, CPU 사양이 일관되지 않음
- batch size, sequence length 등 workload config가 표에 빠져 있는 경우 다수
- 서로 다른 baseline으로 평가한 결과들을 직접 비교할 수 없음 (논문 스스로 인정)
- Switch Transformer 기반 Table 9는 각 시스템이 다른 baseline을 써서 사실상 비교 불가

**코드 공개:**
- (논문) 리포지토리: https://github.com/MoE-Inf/awesome-moe-inference/ (논문 목록은 공개, 각 시스템 코드는 개별 논문에 따름)

---

## 7. 결과 분석

### Expert Offloading에서 이득이 어디서 오는가

(논문) 각 시스템의 핵심 speedup source를 분해하면:

```
Fiddler의 8.2× 이득
└── CPU에서 expert 연산을 직접 수행
    GPU→CPU activation 복사 + CPU 연산 + CPU→GPU 복사
    < GPU 대기 + CPU→GPU expert 로딩
    (activation이 expert weight보다 훨씬 작기 때문)

HOBBIT의 2.3× 이득
├── Cache miss 시 low-precision expert 로딩 (로딩 시간 단축)
├── LRU + LFU + LHU 복합 캐시 정책
└── 중요도 낮은 expert는 INT4로, 높은 것은 FP16으로 동적 결정

MoE-Lightning의 3.5× 이득
└── CPU-GPU-I/O를 동시 파이프라인으로 실행
    (CPU는 연산, GPU는 연산, I/O는 로딩을 동시에)
```

### Prefetching 정확도 90%의 근거

(논문) Mixtral-Offloading, AdapMoE, HOBBIT 등 여러 논문이 공통으로 보고: **adjacent layer gating input의 high similarity** 때문에 현재 layer의 gating input으로 다음 layer의 expert를 예측하면 약 90% 정확도가 달성된다. LLM의 residual stream 구조상 인접 layer 간 activation distribution이 유사하게 유지되는 것이 근본 이유.

---

## 8. 비판적 분석

### 강점

(분석) 이 survey의 가장 큰 기여는 **offloading 시스템들을 동일한 workload(Mixtral-8x7B)로 정규화**하여 비교한 Table 8이다. 기존 논문들은 각자 편의에 맞는 baseline을 골라 직접 비교가 불가능했는데, 이 표는 그 문제를 부분적으로나마 해소한다.

### 한계

(분석) 치명적 약점: **Switch Transformer 기반 Table 9는 정규화 실패** — 시스템마다 baseline이 달라 사실상 비교가 불가능하다. 논문 스스로 이를 인정하지만, 해결책은 제시하지 않는다.

(분석) Hardware-level 섹션이 매우 얇다. MoNDE, FLAME, Duplex 등을 다루지만 각 아키텍처의 실제 성능 수치나 비교가 부실하다. 이 분야는 급속히 발전 중이라 survey 시점에 이미 outdated된 부분이 있을 수 있다.

(분석) **MoH (Mixture of Heads)** 등 head-level MoE에 대한 system-level 함의는 거의 논의되지 않는다. MoH가 head 단위로 더 잘게 더 자주 활성화된다는 점이 offloading/prefetching에 어떤 차이를 만드는지 분석이 없다.

(분석) 에너지 효율에 대한 논의는 Section 6.2.1에 집중되어 있으나, 실제 측정치가 없는 방향 제시 수준이다.

### 숨은 가정

(분석) Prefetching 90% 정확도 주장은 **특정 모델과 workload에서 측정된 것**이다. 모델 크기, expert 수, 도메인이 다를 때의 일반화 가능성은 검증되지 않았다. 특히 expert 수가 매우 많은 DeepSeek-V3(256 experts) 같은 경우 prediction accuracy가 달라질 수 있다.

---

## 9. 관련 연구 맥락

### Survey 내 주요 lineage

```
MoE 기원 (Jacobs et al. 1991) [73]
    ↓
Sparsely-gated MoE (Shazeer et al., ICLR 2017) [160]  ← 현대 MoE의 출발점
    ↓
Switch Transformer (Fedus et al., JMLR 2022) [46]       ← top-1 routing, 단순화
GShard (Lepikhin et al., ICLR 2021) [90]                ← 분산 training 표준
    ↓
DeepSpeed-MoE (Rajbhandari et al., ICML 2022) [144]     ← 가장 많이 비교되는 inference baseline
FastMoE [58] / FasterMoE [59]                           ← 두 번째로 많이 쓰이는 baseline
    ↓
Mixtral-8x7B (Jiang et al., 2024) [75]                 ← offloading 연구의 표준 평가 모델
DeepSeekMoE [30] / DeepSeek-V2 [32] / DeepSeek-V3 [33] ← fine-grained expert + shared expert 주류화
```

(외부 — arXiv) 이 survey와 동시기에 나온 관련 survey: Cai et al., "A Survey on Mixture of Experts in Large Language Models," IEEE TKDE 2025 [14] — architecture 중심. 본 survey는 **inference optimization 특화**로 차별화.

---

## 10. 🎯⭐ 내 연구와의 연결점 · 적용 가능성

### Adjacent-layer Gating 유사성: 예측이 통하는 이유 (심층 정리)

(논문) Mixtral-Offloading, AdapMoE, HOBBIT, EdgeMoE가 공통적으로 관찰한 현상: **인접 layer의 gating input은 높은 similarity를 가진다**. 이는 LLM의 residual connection 구조에서 자연스럽게 발생한다 — $x_{l+1} = x_l + \text{MoE}_l(x_l)$이므로, $x_{l+1}$과 $x_l$의 방향이 크게 다르지 않다. 결과적으로 layer $l$의 router에 입력되는 $x_l$은 layer $l+1$의 router 입력 $x_{l+1}$과 충분히 유사해서, $x_l$을 그대로 layer $l+1$의 router에 통과시켜도 약 90%의 expert 선택이 일치한다.

(분석) 더 정확히는: $x_l$은 "현재 token의 contextualized embedding이 여기까지 쌓인 상태"이고, expert selection의 semantic이 급격히 바뀌지 않는다는 것을 의미한다. 이것이 **Markov 가정**이 적용 가능한 근거다 — layer $l$에서의 expert 선택은 layer $l-1$의 상태(activation + 선택된 expert)로부터 상당 부분 예측 가능하다.

### Prefetch 정책 분류: Cache-based vs Prefetch-based

이 survey의 시스템들을 prefetch 전략 기준으로 분류하면:

| 분류 | 방법 | 대표 시스템 | 예측 근거 |
|---|---|---|---|
| **Static / Profile-based** | calibration set으로 profiling | EdgeMoE, Fiddler (static 부분) | 데이터셋 의존, 환경 변화에 취약 |
| **LRU/LFU Cache-only** | 최근/최빈 사용 기록 | Mixtral-Offloading (LRU), MoE-Infinity (LFU 변형) | 예측 없음, 캐시 관리만 |
| **Adjacent-layer Gating** | 현재 layer 입력 → 다음 layer router 통과 | Mixtral-Offloading (prefetch 부분), HOBBIT, AdapMoE | 90% 정확도, 구현 단순 |
| **Sliding-window MLP Predictor** | 작은 MLP로 k-layer 앞 예측 | ProMoE (window=4~8) | offline 학습 필요, 더 긴 horizon |
| **Full-pass Predictor** | forward pass 시작 시 전체 routing 예측 | ExpertFlow, SiDA (LSTM), Read-ME | 높은 정확도, 복잡도 높음 |
| **Pre-gating (구조 변경)** | 이전 layer 계산 중 다음 expert 미리 결정 | Pre-gated MoE | 모델 구조 수정 필요, 약간의 정확도 손실 |

### 내 Markov Table의 위치

(분석) **내 Markov table 접근은 "Adjacent-layer Gating"과 "Sliding-window MLP Predictor" 사이에 위치한다.** 구체적으로:

- Markov table은 `(현재 layer, 현재 expert 집합) → (다음 layer의 expert 집합 분포)`를 학습하는 구조
- Adjacent-layer gating은 activation을 직접 넘기는 것이고, Markov table은 **선택된 expert 집합 자체를 상태(state)로 쓰는 더 coarse한 표현**
- 장점: activation을 CPU로 넘길 필요 없이 expert index만으로 동작 → overhead 낮음
- 단점: 같은 expert 집합을 선택했더라도 다른 activation일 수 있음 → ProMoE MLP보다 정보량 적음

따라서 분류상: **Cache-based에 가까운 lightweight prefetch-based**, 즉 "history-driven (expert-choice level)" 카테고리로 볼 수 있다. 논문에서 가장 가까운 것은 EdgeMoE의 prediction table이지만, EdgeMoE는 per-layer static table인 반면 Markov table은 dynamic하다.

### Head 단위 vs Expert 단위 차별점

(논문 + 분석) MoH [77] 등 Mixture-of-Heads 구조에서는 attention head를 expert처럼 routing한다. Expert MoE와의 결정적 차이는 **granularity와 activation frequency**다. Expert MoE에서는 top-K(보통 2~8)가 선택되며 expert 하나의 크기가 크다(수백MB). 반면 MoH에서는 head 단위이므로 expert 하나의 크기가 극히 작고(단일 head weight ~수MB 이하), 선택 개수는 전체 head 수에서 k개로 훨씬 많은 head가 맥락에 따라 활성화된다. 이 차이가 offloading/prefetching 관점에서는 중요하다: head MoE는 granularity가 작아 **하나의 cache miss가 미치는 impact가 작고, cache hit ratio를 높이기 쉬우며, prefetch 단위가 작아 빠르게 로딩 가능**하다. 반면 개수가 많아 전체 working set의 크기는 비슷하거나 더 클 수 있다. 결론적으로, head 단위 MoE는 expert 단위 MoE보다 **더 잦은 activation 교체, 더 낮은 per-miss cost, 하지만 더 복잡한 scheduling**을 요구한다. 내 연구에서 MoH를 target으로 한다면 prefetch 정책의 granularity와 window size를 Expert MoE보다 작게, 더 자주 갱신하도록 설계해야 한다.

### Accelerator Simulation 관점 적용

(분석) Accel-sim 기반 simulation에서 MoE workload를 평가할 때 이 survey가 알려주는 핵심 포인트:

- **Expert 선택의 동적성**을 simulation에서 어떻게 모델링할지 결정 필요 — trace-driven이라면 실제 routing decision을 replay해야 함
- Expert loading이 전체 시간의 >80% → GPU compute utilization이 낮게 나오는 것이 당연. MoE는 memory-bandwidth bound 워크로드
- Expert parallelism의 two-pass all-to-all 비용이 simulation에서도 지배적 → interconnect 모델링이 핵심

---

## 11. 핵심 용어 정리

| Term | 정의 |
|---|---|
| **conditional computation** | 입력에 따라 다른 computational path를 밟는 방식 (vs 항상 같은 연산을 하는 dense) |
| **expert parallelism** | 각 device가 전체 expert의 부분집합을 보유하는 분산 방식 |
| **expert offloading** | GPU 메모리 부족 시 expert를 CPU/SSD로 내보내고 필요 시 로딩하는 기법 |
| **expert cache** | GPU 메모리에 상주하는 자주 쓰이는 expert의 subset |
| **LRU / LFU / LHU** | Least Recently Used / Least Frequently Used / Least High Precision Used — 캐시 교체 정책 |
| **adjacent-layer gating** | 인접 layer의 gating input 유사성을 이용해 다음 layer의 expert를 예측하는 것 |
| **pre-gated MoE** | 현재 layer 계산 중 다음 layer의 expert를 미리 결정하도록 구조를 바꾼 방식 |
| **MoH (Mixture of Heads)** | attention head를 expert처럼 routing하는 구조 |
| **sparse-to-dense** | MoE sparse 모델을 knowledge distillation 등으로 dense 모델로 변환 |
| **all-to-all communication** | 분산 MoE에서 토큰을 각 expert의 GPU로 재분배하는 collective operation |
| **load balancing loss** | 특정 expert에 토큰이 집중되는 것을 방지하기 위한 auxiliary training loss |
| **Op/B (Operations per Byte)** | 연산 수 대비 메모리 접근량 — MoE는 낮아서 memory-bandwidth bound |
| **NDP (Near-Data Processing)** | 메모리 근처에 연산 유닛을 배치해 데이터 이동을 줄이는 아키텍처 |
| **shared expert** | 모든 토큰에 항상 적용되는 dense expert (DeepSeekMoE 방식) |
| **top-k routing** | 각 토큰이 상위 k개의 expert를 선택하는 표준 routing 방식 |
| **expert choice routing** | 반대로 각 expert가 상위 k개의 토큰을 선택하는 방식 (고정 bucket size) |

---

## 12. 미해결 질문 / 더 읽을 것

### 논문이 남긴 Open Questions

- 극단적으로 많은 expert(256~512개)에서 prefetching 정확도는 어떻게 변하는가?
- Energy-aware expert placement/scheduling의 실용적 구현 방법은?
- MoE와 speculative decoding을 결합하면 어떤 상호작용이 생기는가?
- Head-level MoE(MoH)의 offloading 최적화는 expert-level과 어떻게 달라야 하는가?
- 표준화된 MoE inference benchmark가 없는 상태에서 fair comparison을 어떻게 달성할 것인가?

### 후속으로 읽을 논문 (직접 연결되는 것 우선)

| 논문 | 이유 |
|---|---|
| **HOBBIT** (Tang et al., arXiv 2411.01433) | 이 survey의 공저자 논문; mixed-precision offloading, LHU 캐시, 토큰-level dynamic precision 선택의 구현 세부사항 |
| **MoE-Infinity** (Xue et al., arXiv 2401.14361) | request-level LFU 기반 prefetch; MoE-Infinity→HOBBIT 순으로 읽으면 evolution이 보임 |
| **ProMoE** (Song et al., arXiv 2410.22134) | sliding-window MLP predictor; Markov table과 직접 비교 가능한 baseline |
| **Mixtral-Offloading** (Eliseev & Mazur, arXiv 2312.17238) | adjacent-layer gating prediction의 원조; LRU 캐시 구현 세부사항 |
| **AdapMoE** (Zhong et al., ICCAD 2024) | Fisher information 기반 expert 중요도 + adaptive cache size; edge device 타겟 |
| **Fiddler** (Kamahori et al., ICLR 2024) | CPU-assist computation; CPU 연산이 GPU 대기보다 빠른 조건 분석 |
| **MoH** (Jin et al., ICML 2042) | head-level MoE의 구조와 라우팅 메커니즘 |
| **Pre-gated MoE** (Hwang et al., ISCA 2024) | 구조 변경으로 prefetch를 강제하는 HW-SW co-design 접근 |
| **ExpertFlow** (He et al., arXiv 2410.17954) | full routing-path prediction; 내 Markov table과 설계 철학 비교용 |
| **DeepSeekMoE** (Dai et al., ACL 2024) | shared expert + fine-grained expert 구조의 원점; 왜 256 expert가 필요한지 |

---

*작성 기준: 논문 본문(ACM Comput. Surv. 2026) 전체 + 공개 arXiv 링크 기반. 외부 검색은 시스템 lineage 확인에 활용.*
