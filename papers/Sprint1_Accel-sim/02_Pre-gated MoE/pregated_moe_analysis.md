# Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference

## 0. 한눈에 보기

- **출처**: ISCA 2024 (ACM/IEEE 51st Annual International Symposium on Computer Architecture) / DOI: 10.1109/ISCA59077.2024.00078 / arXiv preprint: arXiv:2308.12066
- **저자**: Ranggi Hwang (KAIST), Jianyu Wei (USTC / MSR), Shijie Cao, Changho Hwang, Xiaohu Tang, Ting Cao, Mao Yang (Microsoft Research)
- **분류 태그**: #MoE-inference #CPU-offloading #HW-SW-codesign #expert-prefetch #LLM-serving #memory-efficiency
- **TL;DR**: Gate function을 한 블록 앞으로 옮겨 expert selection–execution 간의 데이터 의존성을 완전히 끊어냄으로써, CPU→GPU expert migration 레이턴시를 execution에 완전히 overlap시키는 algorithm–system co-design.
- **핵심 기여**:
  1. **(Algorithm) Pre-gate function**: N번째 MoE 블록이 (N+1)번째 블록에서 사용할 expert activation mask를 미리 결정. 기존 gate의 데이터 의존성(select → execute 직렬화)을 블록 경계 너머로 이동시킴.
  2. **(System) Preemptive expert migration**: pre-gate 결과를 이용해 activated expert만 골라 prefetch하고, 이를 현재 블록 execution과 overlap.
  3. **단일 GPU 배포 가능성**: 전체 MoE parameter를 CPU DDR에 offload, GPU HBM 사용량 GPU-only 대비 평균 ~77% 절감.
  4. **정확도 무손실**: fine-tuning 단계에서만 pre-gate를 학습, pretrain 재실행 없이 적용 가능하며 accuracy drop 미미.
  5. **SSD offload 확장성**: Switch-XXL(395B) 수준 모델까지 단일 노드에서 동작 시연.
- **핵심 직관**: "expert가 무엇이 선택될지 모르기 때문에 migration을 숨길 수 없다"는 근본 제약을, gate를 한 레이어 앞으로 당겨 "한 레이어 뒤의 정답을 미리 알고 있는 상태"로 바꾸면 해결된다.

---

## 1. 문제 정의 & 동기

### 풀려는 문제

MoE 기반 LLM은 동일 FLOPs의 dense 모델에 비해 훨씬 큰 parameter 수를 갖는다(SwitchTransformer: 75× memory vs. T5). 이 대규모 model을 단일 GPU에 올리면 OOM, multi-GPU expert parallelism을 쓰면 sparse activation으로 인해 GPU utilization이 극도로 낮아진다. CPU offload는 이 memory 문제를 해결하는 유일한 low-cost 수단이지만, **migration latency를 숨길 방법이 없다**는 근본 한계가 있었다.

### 기존 접근의 한계 (Gap)

| 접근법 | 핵심 아이디어 | 한계 |
|---|---|---|
| **GPU-only** (multi-GPU, expert parallelism) | 전체 expert를 GPU HBM에 분산 보관 | Sparse activation → 실제 사용 expert가 극소수 → 낮은 GPU utilization; inter-GPU all-to-all 통신 오버헤드; 대규모 모델은 OOM |
| **MoE-OnDemand** (HuggingFace Accelerate) | 모든 expert를 CPU에 offload, activation 후 on-demand fetch | Expert selection 완료 후에만 fetch 시작 → selection–execution 직렬화 → migration latency 그대로 노출 |
| **MoE-Prefetch** (SE-MoE) | 다음 블록의 *전체* expert를 현재 블록 execution과 overlap하여 prefetch | (1) 전체 expert 전송 → expert 수에 비례하는 BW 소비; 256 expert면 1.56% activation인데도 100% fetch. (2) 동시에 현재+다음 expert를 GPU에 올려야 해서 peak GPU memory 여전히 큼 |

두 CPU offload 방식 공통의 근본 문제: **MoE block 내부에서 expert selection(gate)과 expert execution이 데이터 의존성으로 묶여** 있어, migration을 선행 정보 없이 시작할 수 없다.

---

## 2. 핵심 아이디어

기존 gate function은 "현재 블록에 쓸 expert를 현재 블록에서 선택"한다. Pre-gated MoE는 이 역할을 바꿔, **N번째 블록의 gate가 (N+1)번째 블록에 쓸 expert를 선택**하게 만든다(pre-gate). 이로써 (N+1)번째 블록의 입장에서 보면, expert selection이 이미 N번째 블록 시작 시점에 완료되어 있으므로 CPU→GPU migration을 N번째 블록의 execution과 **완전히 parallel**로 진행할 수 있다. migration 크기는 activated expert만큼만이므로(top-1: 전체의 0.8~1.56%), BW 소비도 최소화된다.

통하는 이유: MoE는 layer depth가 쌓일수록 인접 레이어 간 hidden state가 유사해지는 residual structure를 갖는다. N번째 블록의 activation이 (N+1)번째 블록에서 무엇이 선택될지를 충분히 예측할 수 있을 만큼 correlated되어 있다.

---

## 3. 배경 (필요한 것만, 압축)

### MoE 블록의 two-stage 처리 구조

기존 MoE block 실행 순서:
```
input → gate(linear + softmax) → top-k expert 선택 → 선택된 expert FFN 실행 → output
         ↑ 여기서 어떤 expert가 활성화될지 결정됨
```

이 구조에서 "어떤 expert가 필요한지"는 gate 계산이 끝나기 전까지 **알 수 없다**. Gate는 compact MLP로 연산 비용이 낮지만 그 출력에 execution이 data-dependent하게 묶여 있어 migration을 미리 시작할 수 없다.

### 메모리 분포 (논문 Figure 3 기준, SwitchTransformer)

- Expert parameters: 전체 model 용량의 압도적 다수를 차지 (Switch-Large 128E: 105.6 GB total)
- Non-MoE parameters (attention, norms, embeddings): 상대적으로 소량 → GPU HBM에 고정
- Expert layer 1개의 크기 ∝ FFN hidden dim² × 2(up+down) → large expert count일수록 GPU 단독 보관 불가

---

## 4. 제안 방법 상세 🎯

### 4-A. (Algorithm) Pre-gate Function

#### 구조적 변경

기존 MoE block $N$:
$$\text{gate}_N(\mathbf{x}_N) \rightarrow \text{mask}_N \rightarrow \text{Expert}^{(N)}_{\text{mask}_N}(\mathbf{x}_N)$$

Pre-gated MoE block $N$:
$$\underbrace{\text{pre-gate}_N(\mathbf{x}_N)}_{\text{다음 블록용 mask 생성}} \rightarrow \text{mask}_{N+1} \rightarrow \text{Expert}^{(N+1)}_{\text{mask}_{N+1}}(\mathbf{x}_{N+1})$$

즉 N번째 블록의 pre-gate 함수는 **N+1번째 블록에서 활성화할 expert의 binary activation mask를 생성**한다.

#### 첫 번째 및 마지막 블록 처리

Pre-gate 구조에서 두 가지 예외 케이스가 존재한다.

- **첫 번째 MoE 블록**: 이전 블록이 없으므로 pre-gate를 제공해 줄 블록이 없다. 따라서 첫 번째 블록에는 **두 개의 gate function**을 사용:
  - 첫 번째 gate: 현재(첫 번째) 블록에 사용할 expert 선택 (전통적인 gate)
  - 두 번째 gate(pre-gate): 다음(두 번째) 블록에 사용할 expert 미리 선택
  - 이 블록만 selection–execution 직렬화가 발생 (논문에서 유일한 예외로 명시)

- **마지막 MoE 블록**: 다음 블록이 없으므로 pre-gate function이 불필요. Gate function 자체를 사용하지 않음.

#### Decoder iteration 간의 관계

Pre-gate는 **같은 decoder iteration 내부**의 블록들 사이에서만 작동한다. Iteration 경계를 넘어 다음 iteration의 expert를 미리 선택하지 않는다. 즉, 매 decoder step(token 생성)마다 위 구조가 독립적으로 반복된다.

#### Training (Fine-tuning only)

Pre-gate 학습 전략:
```
1. 기존 MoE pretrained weights를 그대로 가져옴 (pretrain 재실행 없음)
2. 모델 구조를 pre-gate 형태로 재구성 (첫/마지막 블록 조정 포함)
3. downstream task fine-tuning을 동일 step 수로 수행
   → batch size: 256 sequences × 256 tokens = 65,536 tokens/batch
   → steps: 2,048 (총 ~2^27 tokens)
   → constant learning rate: 0.0001
4. 기존 MoE fine-tuning과 완전히 동일한 hyperparameter 사용
```

Pre-gate가 "무엇을 학습"하는지: N번째 블록의 입력 activation $\mathbf{x}_N$으로부터 (N+1)번째 블록에서 높은 확률로 선택될 expert의 index를 미리 예측하는 lightweight linear classifier.

### 4-B. (System) Preemptive Expert Migration

#### 실행 타임라인

```
Block N-1:  [Non-MoE] [Expert Exec (N-1)] → pre-gate N 실행 → mask_{N} 확정
                                                    ↓
Block N:    [Non-MoE] [Expert Exec (N)]   ← overlap → [PCIe: fetch Expert(N)]
                          ↑                                        ↑
                    현재 블록 execution           pre-gate N이 미리 결정한
                                                 activated expert만 전송
```

핵심: Pre-gate 계산은 compact MLP라 GPU compute cost가 극히 낮음(PCIe communication-bound 구간). 따라서 pre-gate 완료 후 즉시 PCIe transfer를 시작하면, 다음 블록의 compute가 시작될 때 transfer가 이미 완료되거나 overlap 중인 상태를 만들 수 있다.

#### Peak GPU Memory 수식

$$\text{Peak GPU Mem} = M_{\text{NonMoE}} + \max_{\forall N,\, 0 \le N < B} \left( \sum_{L=N}^{N+1} \text{ActExp}_L \right)$$

여기서:
- $M_{\text{NonMoE}}$: attention layer, normalization, embedding 등 비-MoE parameter 크기 (GPU 상주)
- $\text{ActExp}_L$: $L$번째 MoE 블록에서 실제 활성화된 expert들의 합산 크기
- $B$: 전체 MoE 블록 수

즉 어느 순간이든 GPU에는 **현재 블록 + 다음 블록**의 activated expert만 올라와 있으면 된다. Top-1 activation(SwitchTransformer 기본값)이면 전체 expert 중 2개 블록 × 1 expert만 GPU에 상주.

이 수식이 MoE-OnDemand와 거의 동일한 memory footprint를 보장하는 이유: OnDemand도 activated expert만 fetch하므로 peak memory가 유사. Pre-gated MoE는 OnDemand 대비 0.2% GPU memory 추가 사용(두 블록의 expert를 동시에 올려야 하므로 미미한 차이 발생).

---

## 5. 시스템 · HW 측면

### 메모리 계층 구성 (논문 Figure 4)

```
GPU HBM (80 GB, NVIDIA A100)
  └─ Non-MoE parameters (상시 상주): attention, norms, embeddings
  └─ Activated expert parameters (동적, 2개 블록 분량): pre-gate에 의해 PCIe로 fetch

PCIe Gen4 (32 GB/s bidirectional)
  └─ CPU DDR4 (1.8 TB, AMD EPYC 7V12)
       └─ 전체 MoE expert parameters offload (Switch-Large: ~100+ GB)

(확장) NVMe SSD
  └─ Switch-XXL(395B)처럼 CPU DRAM조차 초과하는 모델용
```

### Compute vs. Communication 분담

| 단계 | 실행 위치 | 비고 |
|---|---|---|
| Non-MoE layer (attention 등) | GPU | 상시 상주 parameter 사용 |
| Pre-gate computation | GPU | Compact MLP, 저연산 → PCIe transfer와 overlap |
| Expert FFN execution | GPU | Activated expert만; compute-bound |
| Expert parameter migration | PCIe (CPU→GPU) | Communication-bound; expert execution과 overlap |

### SW 구현

- 전체 시스템을 **NVIDIA FasterTransformer** (CUDA 기반 production inference library) 위에 구현 (논문)
- GPU-only, MoE-OnDemand, MoE-Prefetch, Pre-gated MoE 모두 동일 베이스 라이브러리 위 구현 → 공정 비교
- C++/Python 구현 (Appendix 기준)
- RTL/합성 없음: 기존 NVIDIA A100 + PCIe Gen4 commodity HW 그대로 사용

### Artifact 공개

- **GitHub**: https://github.com/ranggihwang/Pregated_MoE
- **Zenodo**: https://doi.org/10.5281/zenodo.10976343
- 재현에 필요한 HW: GPU 40GB 이상 + CPU 128GB 이상 + disk 100GB 이상

---

## 6. 🎯⭐ 실험 셋업 & 재현 관점

### HW Platform

| 항목 | 사양 |
|---|---|
| CPU | AMD EPYC 7V12, 64-core, 1.8 TB DDR4 |
| GPU | NVIDIA A100 (80 GB HBM), **단일** GPU |
| PCIe | Gen4, 32 GB/s |
| 소프트웨어 | NVIDIA FasterTransformer (CUDA), C++/Python |

### Model Configurations

(논문 Table I, SwitchTransformer 기준)

| Model | Experts | Layers | Parameters | Capacity |
|---|---|---|---|---|
| Switch-Base-8 | 8 | 12 | 0.7 B | 2.8 GB |
| Switch-Base-64 | 64 | 12 | 3.8 B | 15.2 GB |
| Switch-Base-128 | 128 | 12 | 7.5 B | 30.0 GB |
| Switch-Base-256 | 256 | 12 | ~15 B | ~60 GB (추정) |
| Switch-Large-128 | 128 | 24 | 26.4 B | 105.6 GB |
| Switch-XXL (SSD 실험) | - | - | 395 B | 217 GB (quantized) |

- Switch-Large-128은 105.6 GB → A100 80 GB에서 GPU-only는 OOM
- Switch-XXL은 quantization 적용 후 217 GB → CPU DRAM에도 맞지 않아 SSD offload 필요

### Baselines

| Baseline | 구현 출처 | 핵심 동작 |
|---|---|---|
| **GPU-only** | 논문 자체 구현 (FasterTransformer) | 전체 parameter GPU 상주, 오라클 상한 (OOM for Large) |
| **MoE-OnDemand** | HuggingFace Accelerate [15] | expert 전체 CPU offload, selection 후 on-demand fetch |
| **MoE-Prefetch** | SE-MoE [38] | expert 전체 CPU offload, 다음 블록 전체 expert prefetch |

> (외부 확인) SE-MoE(arXiv:2205.10034)는 Baidu가 2022년 공개한 distributed MoE 시스템으로, training의 "prefetch-all" 개념을 inference에 적용. HuggingFace Accelerate는 MoE-OnDemand에 해당하는 on-demand fetch를 제공하는 오픈소스 라이브러리.

### Datasets & Metrics

| Task | Dataset | Metric |
|---|---|---|
| Summarization | Xsum [23] | Rouge-1 (R1), Rouge-2 (R2) |
| QA (closed-book) | CB Web QA [2] | ExactMatch, F1 |
| QA (closed-book) | SQuAD [32] | ExactMatch, F1 |

- Performance 측정은 SQuAD fine-tuned 모델로 수행 ("end-to-end inference performance is less sensitive to downstream task")
- Accuracy 측정은 모든 downstream task 포함

### Fine-tuning 하이퍼파라미터

| 항목 | 값 |
|---|---|
| Batch size | 256 sequences |
| Sequence length | 256 tokens |
| Fine-tuning steps | 2,048 |
| Total tokens | ~$2^{27}$ |
| Learning rate | 0.0001 (constant) |
| Base weights | HuggingFace 공개 pretrained SwitchTransformer |

### Ablation Study 구성

1. **Pre-gating activation level N**: N=1(default), N=2, N=3 — N이 커질수록 accuracy 저하 예상
2. **Number of activated experts**: 1~64 experts (1.56%~100% activation) — sparsity 증가 시 CPU offload 이점 감소
3. **Expert caching**: Pre-gated MoE + MoE-OnDemand 각각에 LIFO/LFU/LRU 캐시 정책, 1%/10%/20% cache ratio → caching이 OnDemand에 더 효과적임을 확인
4. **SSD offloading**: Switch-XXL 모델 대상 SSD 기반 offload 비교

### 🔎 재현 체크리스트

| 항목 | 상태 |
|---|---|
| 코드 공개 | ✅ GitHub + Zenodo |
| Pre-trained weight | ✅ HuggingFace 공개 SwitchTransformer 사용 |
| HW spec | ✅ A100 80GB + AMD EPYC + PCIe Gen4 32 GB/s |
| Fine-tuning hyperparameter | ✅ 전부 명시 |
| Inference batch size | ✅ batch size 1 (single-token decoding) |
| PCIe BW 측정 방법 | ⚠️ 논문에 명시 없음 — 실측값인지 명목값인지 불명확 |
| Pre-gate 구조 (hidden dim 등) | ⚠️ "compact MLP"라고만 기재, 정확한 architecture 미공개 |
| Expert caching LRU/LFU 구현 상세 | ⚠️ 자체 구현이나 라이브러리 기반인지 불명확 |
| SSD 모델(Switch-XXL) weight 공개 여부 | ⚠️ 논문에 명시 없음 |
| Quantization 방식 (Switch-XXL 217GB) | ⚠️ "quantization 적용 후 217 GB"라고만 언급, 방식 미기술 |

---

## 7. 결과 분석

### 7-A. MoE Block Latency (논문 Figure 10)

(y축: GPU-only 대비 normalized, log scale)

| Model | GPU-only | Pre-gated MoE | MoE-OnDemand | MoE-Prefetch |
|---|---|---|---|---|
| Switch-Base-8E | 1× | 1.2× | 1.2× | 1.2× |
| Switch-Base-64E | 1× | 1.2× | 2.0× | 7× (추정) |
| Switch-Base-128E | 1× | 1.2× | 1.9× | ~54× (추정) |
| Switch-Large-128E | OOM | 1× (base) | 1.9× | 125× |

> (논문) Switch-Base 전 구성에서 Pre-gated MoE는 GPU-only 대비 평균 19% latency overhead. MoE-OnDemand 대비 평균 1.7× (최대 1.9×), MoE-Prefetch 대비 평균 42× (최대 125×) 개선.

MoE-Prefetch가 expert 수에 따라 폭발적으로 악화되는 이유: 64E → 128E → 256E로 expert 수가 늘수록, 다음 블록의 **모든** expert를 fetch해야 하므로 transfer 시간이 선형 증가. SwitchTransformer Top-1 activation 기준 실제 쓰이는 expert는 1개지만 나머지 127개도 모두 전송.

### 7-B. End-to-End Throughput (논문 Figure 11)

| Model | GPU-only | Pre-gated MoE | MoE-OnDemand | MoE-Prefetch |
|---|---|---|---|---|
| Switch-Base avg | ~137 tok/s | **111 tok/s** | ~74 tok/s | ~4 tok/s |
| Switch-Large-128E | OOM | **42 tok/s** | ~26 tok/s | ~0.8 tok/s |

> (논문) Pre-gated MoE는 Switch-Base에서 GPU-only의 81% throughput 달성. MoE-OnDemand 대비 평균 1.5× (최대 1.6×), MoE-Prefetch 대비 평균 27× (최대 55×).

### 7-C. Peak GPU Memory (논문 Figure 12)

| Model | GPU-only | Pre-gated MoE | MoE-OnDemand | MoE-Prefetch |
|---|---|---|---|---|
| Switch-Base-8E | 1× | ~0.1× | ~0.1× | ~0.35× |
| Switch-Base-256E | 1× | ~0.04× | ~0.04× | ~0.5× |
| Switch-Large-128E | OOM | ~0.23×\* | ~0.23×\* | 1×\* |

\* Switch-Large는 GPU-only OOM이므로 MoE-Prefetch를 1×로 normalize

> (논문) Pre-gated MoE는 GPU-only 대비 평균 **23%** GPU memory 사용. Memory-optimal MoE-OnDemand 대비 0.2% 추가 사용에 불과.

### 7-D. Model Accuracy (논문 Table II)

| Model | Task/Metric | Baseline MoE | Pre-gated MoE | 변화 |
|---|---|---|---|---|
| Base-8 | Xsum R1 | 34.6 | 34.7 | +0.1 |
| Base-8 | CB Web QA ExactMatch | 26.0 | 28.2 | **+2.2** |
| Base-8 | SQuAD ExactMatch | 77.4 | 78.2 | +0.8 |
| Base-128 | Xsum R1 | 38.1 | 38.0 | -0.1 |
| Base-128 | CB Web QA ExactMatch | 27.4 | 25.8 | -1.6 |
| Base-128 | SQuAD ExactMatch | 81.7 | 82.2 | +0.5 |
| Large-128 | Xsum R1 | 40.2 | 40.1 | -0.1 |
| Large-128 | CB Web QA ExactMatch | 31.0 | 30.5 | -0.5 |
| Large-128 | SQuAD F1 | 90.1 | 90.2 | +0.1 |

**🎯 오예측(misprediction) 관련 핵심 관찰**: 논문은 accuracy 손실이 미미하다고 주장하지만, **misprediction penalty를 직접 정량화하는 실험이 없다**. 즉:
- Pre-gate가 틀렸을 때 어떤 일이 벌어지는가? → 논문에 명시 없음.
- (분석) Pre-gate 출력은 **deterministic activation mask**로 설계됨. Pre-gate의 top-k 선택이 실제 gate와 다를 경우, 그 차이가 곧 model accuracy degradation으로 나타난다. 즉 misprediction penalty = accuracy drop 그 자체이며, 별도의 system-level penalty(추가 migration 등)는 존재하지 않는다. Pre-gate 예측이 틀렸다고 해서 "정확한 expert를 다시 fetch"하는 fallback 메커니즘은 구현되어 있지 않다.
- 이는 speculative decoding의 "reject and reroll"과 달리, **틀린 채로 계산을 진행**하는 구조임. 정확도 손실이 발생하더라도 latency penalty는 없다는 점이 설계의 트레이드오프.

### 7-E. Activation Level Ablation (논문 Figure 13, Switch-Base-8E, SQuAD)

| Pre-gate Activation Level | ExactMatch | F1 |
|---|---|---|
| Conventional MoE (N=0) | ~77.4 | ~85.8 |
| Pre-gated (N=1, default) | ~78.2 | ~86.0 |
| Pre-gated (N=2) | ~77.8 | ~85.9 |
| Pre-gated (N=3) | ~77.3 | ~85.6 |

N=1이 N=0(기존 MoE)보다 오히려 정확도가 약간 높은 경우가 있음 → (분석) fine-tuning 과정에서 pre-gate가 추가 regularization 역할을 할 가능성이 있으나 논문은 상세 분석을 future work로 남김.

### 7-F. Expert Caching 효과 (논문 Figure 15, Switch-Large-128E)

| 시스템 | Cache 없음 | 최적 cache(1~20%) |
|---|---|---|
| Pre-gated MoE | 1.0× (baseline) | ~1.1~1.2× |
| MoE-OnDemand | ~0.6× | ~0.7~0.9× |

Caching은 OnDemand에 더 큰 상대적 이득을 줌. Pre-gated MoE는 이미 migration을 대부분 숨기고 있어 caching의 추가 이득이 제한적.

### 7-G. 이득이 실제로 어디서 오는가

| 이득 항목 | 메커니즘 | 기여도 |
|---|---|---|
| Latency 감소 vs. MoE-OnDemand | selection–execution 직렬화 제거 → migration overlap | 주된 기여 (1.7× 평균) |
| Latency 감소 vs. MoE-Prefetch | activated expert만 fetch → transfer 크기 대폭 감소 | 압도적 기여 (42× 평균) |
| GPU memory 절감 vs. GPU-only | 전체 expert를 CPU에 offload | 4.2× (= 1/0.23) |
| Throughput vs. GPU-only gap | PCIe transfer가 hidden되지 못하는 첫 블록 + 잔여 PCIe overhead | ~19% gap의 근원 |

---

## 8. 비판적 분석

### 강점

1. **근본 제약 해결**: MoE-offload가 공통으로 가진 "selection 완료 전 migration 불가"라는 데이터 의존성을 알고리즘 레벨에서 끊은 점이 핵심. System optimization만으로는 불가능한 해결책.
2. **Pretrain 재실행 불필요**: Fine-tuning 단계만으로 pre-gate 학습. 실용적으로 중요한 포인트.
3. **Memory 거의 OnDemand 수준, 성능은 GPU-only에 근접**: 두 극단의 중간이 아닌, 두 극단 모두에 근접.
4. **코드 공개**: ISCA 논문으로는 드물게 complete artifact 공개 (Zenodo + GitHub).

### 한계 및 숨은 가정

1. **(분석) Single-batch(BS=1) 가정의 제한성**: 모든 성능 실험이 batch size 1 기준. Production serving에서 throughput 극대화를 위해 batching(BS > 1)을 쓸 경우, PCIe transfer와 compute의 overlap 비율이 달라지고 MoE-Prefetch와의 격차가 줄어들 수 있음. 논문은 "real-world production ML serving systems are optimized for a batch size of 1"[9][10][34]를 근거로 정당화하나, LLM serving의 continuous batching 맥락에서는 재검토 필요.

2. **(분석) Misprediction에 대한 정량화 부재**: Pre-gate가 실제 gate와 얼마나 자주 다른 expert를 선택하는지(misprediction rate) 측정값이 없음. Accuracy table(Table II)이 간접 지표이나, "prediction accuracy = ?" 수치가 직접 제시되지 않음. (외부) 후속 연구들(HOBBIT [arXiv:2411.01433] 등)은 유사한 hidden-state 기반 예측이 ~90% 정확도를 달성한다고 보고하지만, Pre-gated MoE의 정확 수치는 논문 내 미기재.

3. **(분석) 첫 블록 직렬화**: 모든 decoder iteration에서 첫 번째 MoE 블록은 여전히 selection–execution이 직렬화됨. 모델의 총 MoE 블록 수가 적을수록(예: 12 layers Switch-Base) 이 overhead의 비중이 커짐.

4. **(분석) SwitchTransformer(Top-1) 특화**: Top-1 expert activation은 극단적으로 sparse한 케이스. Mixtral (Top-2 of 8)처럼 expert 수가 적고 activation 비율이 높은 모델에서는 MoE-Prefetch와의 격차가 줄어든다. Figure 14의 sensitivity study가 이를 보여주지만, 실세계 모델(Mixtral-8x7B, Mixtral-8x22B, Qwen-MoE 등)에 대한 직접 평가 없음.

5. **(분석) PCIe BW 32 GB/s 가정**: PCIe Gen4 × 16 이론치. 실제 가용 BW는 시스템 구성에 따라 20~25 GB/s 수준이 일반적. 논문에서 실측 PCIe BW를 보고하지 않아, 결과 재현 시 BW 차이가 성능 차이를 만들 수 있음.

6. **(분석) 모델 다양성 부재**: SwitchTransformer encoder-decoder 구조만 평가. Decoder-only 모델(Mixtral, DeepSeek-MoE 등) — 현재 LLM serving의 주류 — 에 대한 적용 검증 없음.

### 빠진 비교

- **FlexGen** [arXiv:2303.06182 관련]: 단일 GPU 고throughput inference를 위한 tensor-level offload. MoE-specific은 아니지만 단일 GPU 서빙 맥락에서 비교 가치 있음.
- **MoE-Infinity** [Xue et al. 2024]: activation-aware prefetch + caching, Pre-gated MoE의 직접 후속/경쟁 연구. (외부) ISCA 2024 이후 등장한 work로, 논문에 포함되지 않은 것은 시간적 이유.
- **Expert quantization 기반 접근**: INT4/INT8 quantization으로 expert size를 줄이면 prefetch-all의 BW 문제가 완화됨. 이와의 결합이나 비교 없음.

---

## 9. 관련 연구 맥락 (웹조사)

### Lineage

```
Shazeer et al. (ICLR 2017)  ← MoE 원조, top-k sparse gating
        ↓
Fedus et al. (JMLR 2022)    ← SwitchTransformer, Top-1 gating, 논문의 기본 모델
        ↓
Huang et al. (arXiv 2303.06182, 2023)  ← MoE deployment 특성 분석, Expert Buffering (LIFO), Dynamic gating
SE-MoE (arXiv 2205.10034, 2022)        ← Prefetch-all for inference, 이 논문의 MoE-Prefetch baseline
HuggingFace Accelerate (2022)          ← MoE-OnDemand baseline
        ↓
Pre-gated MoE (ISCA 2024)   ← 이 논문
        ↓
후속/경쟁 연구:
  MoE-Infinity (Xue et al., 2024)  ← activation-aware prefetch + LRU-like caching, Mixtral 대상
  HOBBIT (arXiv 2411.01433, 2024)  ← mixed precision expert offload + 예측기
  ExpertFlow (arXiv 2410.17954, 2024) ← expert activation matrix 기반 prefetch
  BuddyMoE (arXiv 2511.10054)     ← expert redundancy 활용 offload 가속
```

### Novelty의 실제 위치

(외부, 웹조사 기반) Pre-gated MoE의 핵심 아이디어 — "gate를 한 레이어 앞으로 당겨 prediction 확정"은 2024년 당시 ISCA 수준에서 독창적이었음. 그러나 동시기/후속 연구들은 유사한 아이디어를 **모델 구조 변경 없이** 구현하려는 방향으로 발전:

- (외부) **MoE-Infinity**, **Mixtral-Offloading**: 현재 레이어의 gate input (hidden state)를 다음 레이어 예측기에 직접 사용 → 모델 weight 변경/fine-tuning 없이 약 90% 예측 정확도 달성.
- (외부) **HOBBIT**, **AdapMoE**: 역시 hidden state 기반 예측 + mixed precision quantization 결합.
- (분석) Pre-gated MoE의 약점(모델 구조 변경 + fine-tuning 필요)을 회피하는 방향으로 field가 진화. 그러나 Pre-gated MoE는 **deterministic mask**를 보장한다는 점(예측이 아닌 확정)에서 다른 접근법과 근본적으로 다름.

---

## 10. 🎯⭐ 내 연구와의 연결점 · 적용 가능성

> 연구 맥락: memory-constrained MoE/MoH offloading & prefetch, Accel-sim 기반 accelerator simulation, INT4 기반 근사, KV cache / attention / memory hierarchy 최적화.

### 10-A. 직접 활용 가능한 아이디어/기법

**1. Pre-gate = "확정형 prefetch 결정기"**
Pre-gate의 핵심 가치는 prefetch 결정을 **확정적(deterministic)**으로 만든다는 점. 내 연구에서 MoH(Mixture-of-Heads) 또는 MoE 레이어의 expert/head를 offload할 때, 어느 expert가 필요할지 미리 알면 prefetch와 현재 computation을 overlap할 수 있다. 내가 INT4 Markov 모델(인접 레이어 간 expert activation 전이 확률)로 예측할 경우:

  - **Pre-gated MoE 방식**: 모델 구조에 경량 predictor 삽입 → 확정된 mask 획득 → 오예측이 accuracy loss로 직결
  - **Stochastic 예측 방식** (내 INT4 Markov 접근): 확률 기반 top-k prefetch → 오예측 시 fallback fetch 필요 → latency penalty 있지만 accuracy 보존
  - (분석) 두 접근은 "accuracy vs. system complexity" 트레이드오프. 내 연구에서 "INT4 Markov 대체 시 품질 보존"을 측정하려면, Pre-gated MoE의 Table II 형식(동일 metric, 동일 dataset, baseline vs. proposed 직접 비교)을 참조할 것.

**2. Eq. 1의 Peak GPU Memory 수식**
$$\text{Peak GPU Mem} = M_{\text{NonMoE}} + \max_{\forall N} \left( \sum_{L=N}^{N+1} \text{ActExp}_L \right)$$
이 공식은 offload 시스템의 memory 상한 분석에 바로 사용 가능. 내 accelerator simulation에서 on-chip SRAM budget 계산이나 HBM vs. DRAM 분할 결정에 동일한 프레임을 적용할 수 있음.

**3. Execution timeline overlap 분석 방법 (Figure 9 스타일)**
Compute(green) vs. Communication(blue) 구간을 time-series bar로 시각화하는 Figure 9 방식은 내 논문의 prefetch benefit 제시에 그대로 차용 가능한 표현 방식.

### 10-B. Baseline으로 삼을 수 있는 점

| 항목 | 활용 |
|---|---|
| **MoE-OnDemand** (HuggingFace Accelerate) | 가장 단순한 fetch-on-demand baseline; 내 시스템과 비교의 lower bound |
| **Pre-gated MoE** 자체 | 구조 변경 없는 stochastic prefetch를 제안한다면 Pre-gated MoE가 "구조 변경이 필요한 upper bound" baseline |
| **GPU-only** | Memory 제약이 없는 oracular upper bound; throughput/latency upper bound |
| **MoE-Prefetch** | "naive prefetch-all"의 worst case; 내 예측 기반 selective prefetch의 상대적 이점 계산에 활용 |

### 10-C. Offloading / Prefetch / HW-SW 코디자인 관점의 구체적 응용

**INT4 quantized expert + Pre-gate 결합 가능성**:
- Expert parameter를 INT4로 quantize하면 크기가 4×~8× 감소. Pre-gate의 prefetch 대상 크기가 줄어들어 PCIe overlap 가능성 증가.
- (분석) 반면 Pre-gated MoE 논문에서 Switch-XXL 평가 시 "quantization 적용"을 언급했지만 세부 방식 미기술. INT4 quantized expert prefetch 시 dequantization overhead가 GPU에서 발생하는 latency를 고려해야 함.

**Accel-sim 기반 시뮬레이션 연계**:
- Pre-gated MoE는 실 HW(A100 + PCIe)에서의 wall-clock latency를 측정. Accel-sim을 쓸 경우, PCIe transfer를 memory latency model로 추상화하고, 첫 블록의 직렬화 overhead + 이후 블록의 overlap 비율을 파라미터로 모델링해야 함.
- MoE-block 단위 실행 타임라인(Figure 9)을 Accel-sim의 warp-level timeline에 매핑하는 방법론적 참고 가능.

**MoH(Mixture-of-Heads) 적용 가능성**:
- Pre-gate 개념은 MoE에 국한되지 않음. MoH 구조에서 head-wise offload를 할 경우, 현재 레이어의 attention head output으로 다음 레이어의 활성 head를 미리 예측하는 pre-gate 유사 메커니즘을 설계할 수 있음.
- MoH는 MoE보다 top-k 비율이 높을 수 있어(예: 8 heads 중 4 활성), prefetch 크기 대비 이득 비율 분석이 필요.

### 10-D. 충돌하거나 재검토가 필요한 가정

1. **"Accuracy drop은 미미하다" 주장의 범위**: Table II가 SwitchTransformer만 대상. INT4 quantized expert + pre-gate를 결합하면 quantization error + prediction error가 누적될 수 있음. 내 설계에서는 둘의 상호작용을 별도 측정해야 함.
2. **Batch size 1 가정**: 내 연구가 throughput-optimized serving(높은 배치)을 가정한다면, Pre-gated MoE의 결과를 직접 인용하기 어려움. Batch size가 커질수록 PCIe transfer가 더 큰 compute와 overlap되어 이득이 증가할 수도 있고, 반대로 memory pressure가 늘어 trade-off가 달라질 수도 있음.
3. **Deterministic mask의 훈련 의존성**: 내 연구에서 "사전 훈련된 MoE 가중치를 그대로 쓰면서 pre-gate만 붙이는" 방식을 채택할 경우, fine-tuning 없이 zero-shot으로 pre-gate를 쓸 수 없음. Fine-tuning 비용이 허용되는지 연구 설정 확인 필요.

---

## 11. 핵심 용어 정리

| 영어 용어 | 정의 (이 논문 맥락) |
|---|---|
| **pre-gate function** | N번째 MoE 블록이 (N+1)번째 블록에 사용할 expert activation mask를 미리 생성하는 경량 MLP; 기존 gate function의 timing만 한 블록 앞으로 이동 |
| **activation mask** | 어떤 expert가 활성화되어 실행될지를 나타내는 binary vector; pre-gate의 출력 |
| **MoE-OnDemand** | expert 전체를 CPU에 offload하고 gate selection 완료 후 activated expert를 on-demand로 fetch하는 방식 (HuggingFace Accelerate 구현) |
| **MoE-Prefetch** | expert 전체를 CPU에 offload하고 현재 블록 execution 동안 다음 블록의 전체 expert를 prefetch하는 방식 (SE-MoE 구현) |
| **preemptive expert migration** | pre-gate가 결정한 activated expert만을, 현재 블록 execution과 overlap하여 CPU→GPU로 미리 이동시키는 동작 |
| **expert parallelism** | MoE expert parameter를 여러 GPU에 분산 저장하고, 각 GPU가 담당 expert에 대한 연산을 수행하는 병렬화 전략 |
| **fetch-on-demand** | 필요 시점에 parameter를 memory hierarchy에서 끌어오는 방식; latency가 노출됨 |
| **hot expert** | inference 중 빈번히 활성화되는 expert; caching 대상 |
| **LIFO/LFU/LRU** | Last-In-First-Out / Least-Frequently-Used / Least-Recently-Used cache replacement policy |
| **Switch-XXL** | Switch-Large와 동일 구조이나 feature dimension과 head 수를 4× 증가시킨 대형 변형; 395B parameter, quantized 217 GB |
| **TCO (Total Cost of Ownership)** | 서버 배포 총비용; GPU 수 × 단가 + 운영비 |
| **FLOPs-equivalent** | 동일 추론 연산량 기준으로 비교하는 dense 모델 |
| **capacity (GB)** | 모델을 FP32로 저장했을 때의 총 용량 (parameter count × 4 bytes) |

---

## 12. 미해결 질문 / 더 읽을 것

### 논문이 남긴 Open Questions

1. **Pre-gate misprediction rate 정량화**: Pre-gate가 정확한 expert를 예측하는 비율은? Layer별로 다른가? Token type에 따라 다른가?
2. **N > 1 pre-gate의 시스템 활용**: N=2 이상 pre-gate는 accuracy 측면에서 열등하지만, SSD offload처럼 transfer latency가 매우 긴 환경에서는 더 먼 미래를 미리 prefetch하는 이득이 accuracy 손실을 상쇄할 수 있는가?
3. **Decoder-only 모델 적용**: Mixtral, DeepSeek-MoE, Qwen-MoE 등 decoder-only 구조에서 동일한 방식이 유효한가?
4. **Batch size > 1 시나리오**: Continuous batching 환경에서의 성능 특성.
5. **Fine-tuning 없는 zero-shot pre-gate**: Hidden state similarity를 직접 활용해 fine-tuning 없이 pre-gate를 구성할 수 있는가? (이것이 후속 연구들의 방향)

### 더 읽을 논문 (웹조사 기반)

| 논문 | 이유 |
|---|---|
| **MoE-Infinity** (Xue et al., 2024) [arXiv:2401.15016] | Pre-gated MoE의 가장 직접적인 후속 경쟁 — fine-tuning 없이 activation-aware prefetch + caching. Mixtral 대상 평가 포함. |
| **HOBBIT** (arXiv:2411.01433, 2024) | Mixed precision expert offload + hidden-state 기반 예측기. Pre-gated MoE와 동일 문제를 다른 각도에서 해결. |
| **ExpertFlow** (arXiv:2410.17954, 2024) | Expert Activation Matrix 기반 prefetch + Mixtral 평가. Prediction 방식 비교에 유용. |
| **Mixtral-Offloading** (Eliseev & Mazur, 2023) [arXiv:2312.17238] | MoE-offload 실용 구현 + quantization 결합. Pre-gated MoE가 언급한 "hidden state 기반 예측 ~90% accuracy"의 출처. |
| **Towards MoE Deployment** (Huang et al., 2303.06182) | Expert caching(LIFO), Dynamic gating의 원류. 이 논문에서 [14]로 인용. |
| **SE-MoE** (Shen et al., 2205.10034) | MoE-Prefetch baseline의 원 논문. Prefetch-all의 trade-off 이해에 필수. |
| **DeepSeek-MoE** / **Qwen-MoE** | Top-k of 64 expert 형태의 현대적 MoE 구조에 Pre-gate 아이디어가 어떻게 scale되는지 확인하기 위해 읽을 것. |
| **BuddyMoE** (arXiv:2511.10054, 2025) | Expert redundancy를 활용한 offload 가속. Pre-gated MoE 이후 field의 흐름 파악. |
