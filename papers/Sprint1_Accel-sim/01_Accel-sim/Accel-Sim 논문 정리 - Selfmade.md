# Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling
## 0. 논문 한눈에 보기

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
## 1. 논문 핵심 사항
### 1.1 해결하고자한 문제
1. **기능적 문제**: 매년 새 mISA가 등장하는데 이에 맞춰 기능 모델을 유지하는 비용이 너무 크다. cuDNN 같은 hand-tuned SASS 바이너리는 PTX 표현 자체가 없어 기존 simulator에서 아예 실행 불가한 경우도 있음

2. **정확도 문제**: Volta QV100에서 GPGPU-Sim 3.x는 cycle error 94%, correlation 0.87이다(논문). 이 수준의 오차는 "어떤 아이디어가 효과적인가"를 잘못 판단하게 만듦

---
### 1.2  핵심 아이디어
고정된 함수 모델을 직접 구현하는 대신 Trace-driven시뮬레이션을 채택한 시뮬레이션 프레임워크인 Accel-Sim을 제안

SASS 트레이스를 직접 입력으로 받아 시뮬레이션함으로써, 
functional model 구현에 대한 의존도를 낮추고 
performance model의 정확도와 유연성을 높이는 것

대신 accuracy는 성능 모델(GPGPU-Sim 4.0)의 상세도가 결정하므로, 
저자들은 memory system을 중심으로 현대 GPU 아키텍처를 역공학하여 반영

즉, 실제 하드웨어가 실행하는 코드를 그대로 재현함으로써 모델링의 시작점인 'Baseline'을 실제 하드웨어와 일치시키는 데 중점

---
### 1.3 문제를 해결한 방법
![[Pasted image 20260618163100.png]]

NVIDIA의 NVBit을 활용한 트레이스 생성기와 PTX 기반의 에뮬레이션 모드를 모두 지원하는 'Ambidextrous(양손잡이형) 프론트엔드'를 구축

또한, 마이크로벤치마크를 통해 하드웨어의 L1/L2 캐시 및 메모리 시스템의 파라미터를 자동으로 추론하고 구성하는 튜닝 인프라를 도입

---
### 1.4 Contribution이 뭔지
- 첫째, 최신 CUDA 바이너리 및 폐쇄형 라이브러리(cuDNN 등)를 시뮬레이션할 수 있는 **확장 가능한 프레임워크를 제공**하여 기존 오픈소스 시뮬레이터의 한계를 극복

- 둘째, Kepler에서 Turing 세대까지의 GPU를 모델링하며 성능 모델의 정확도를 대폭 향상해 기존 대비 **Cycle error를 79% 포인트 감소**시켰습니다. 

- 셋째, 하**드웨어 검증을 자동화하는 일련의 도구(Microbenchmarks, Tuner, Correlator)를 공개**하여 향후 연구자들이 새로운 하드웨어 설계를 빠르게 검증할 수 있는 표준적인 방법론을 제시

---
## 2. 나한테 필요한 정보
### 2.1 논문에서 설명한 내용
#### 2.1.1 모드가 두 개지요
> Our new frontend supports both vISA (PTX) execution-driven and mISA (SASS) trace-driven simulation

- Accel-Sim은 사용자가 필요에 따라 execution, trace-driven 모드를 선택할 수 있도록 설계되어 있음
- 데이터 값에 의존적인 복잡한 시뮬레이션이나 동적인 제어 흐름을 정확히 보려면 Execution-driven mode를 사용
- 성능 분석 속도를 높이거나 SASS와 같이 복잡한 기계어(mISA) 수준의 분석이 필요할 때는 Trace-driven mode를 활용
	- Trace-driven 모드에서는, mISA가 1:1 대응되는 SASS 명령어를 갖는 ISA에 비의존적인 중간표현으로 바뀌어서 사용됨
	- Trace-driven 모드에서는, 기능적 에뮬레이션을 생략하고 핵심 작업만 선별적으로 시뮬레이션함으로써, 기존 시뮬레이터 대비 월등한 속도와 함께 최신 GPU의 SASS 코드까지 정밀하게 모사할 수 있는 효율성을 확보

#### 2.1.2 To add a new specialized unit 하려면
> To add a new specialized unit, the user declares the new unit in the configuration file and maps the machine ISA op codes that use this unit in the ISA def file, as described in Figure 1.

- Config에 새로운 Unit은 선언해야됨
- ISA Def 파일에서 이 유닛을 사용하는 machine ISA op code를 매핑해야됨

> If execution-driven PTX support is required, then the user must also implement the code to emulate the new instruction’s functionality

- Execution-driven의 지원이 필요한 경우, 에뮬레이션하는 코드도 직접 작성해야됨
- 그럼 나의 경우는 어떻게 되는거임?
	- 단순히 하드웨어 구조(캐시)를 추가하는 것이냐, 아니면 새로운 ISA(명령어) 기능을 추가하는 것이냐"에 따라 달라짐
		- 얘를 들어,새롭게 추가한 SRAM 캐시를 제어하기 위한 특수 명령어(예: 캐시 강제 플러시, 특정 영역으로의 데이터 프리페치 등)가 PTX(Virtual ISA) 수준에서 새롭게 정의되어야 한다면, GPGPU-Sim 4.0 내부의 Functional Simulator가 해당 PTX 명령어가 무엇을 의미하는지 해석(Emulation)할 수 있도록 코드를 추가
	- 에뮬 수정이 GPGPU-sim 4.0 수정을 의미하는 것인가!?
		- 즉, 에뮬레이터 수정은 GPGPU-Sim 4.0의 기능적 시뮬레이션 로직을 건드리는 것을 의미
- 요약
	- 단순히 하드웨어의 구성(Configuration)을 바꾸어 캐시를 추가하는 것이라면, Performance Model만 수정하면 되므로 에뮬레이터 수정은 불필요
	- 반면, 새로운 하드웨어 기능을 활용하기 위해 새로운 명령어(ISA)를 도입하거나 기존 명령어의 동작을 PTX 수준에서 재정의해야 한다면, GPGPU-Sim 4.0의 Functional Simulator(에뮬레이터) 수정이 필수적

#### 2.1.3 다양한 Config에서마이크로벤치를 제공하지만 전부 제공하는 건 아님
> For other parameters that cannot be directly determined by our microbenchmarks (such as warp scheduling, memory scheduling, the L2 cache interleaving granularity and the L2 cache hashing function), Accel-Sim simulates each possible combination of these four parameters on a set of memory

- warp scheduling, memory scheduling, the L2 cache interleaving granularity and the L2 cache hashing function 같은건 directly 알 방법이 없다고 함.
	- 이러한 요소들은 GPU의 핵심적인 제어 로직이나 고도로 최적화된 하드웨어 배선 수준에서 결정되는 정책
	-  해당 로직이 정확히 어떤 알고리즘으로 돌아가는지 직접적인 수치(Latency나 Bandwidth 등)를 추출하기 어려움
	- 정 단일 테스트만으로 하드웨어 내부의 스케줄링 알고리즘을 역공학(Reverse Engineering)하여 알아내기는 매우 어려움
- 해결 방법
	- 탐색 기반 추정
		- 직접적인 측정값이 없기 때문에, 발생 가능한 ombination을 시뮬레이터에서 미리 설정하고 시뮬레이션을 수행
	- 상관관계 분석 
		- 시뮬레이션된 각 조합의 결과값과 실제 하드웨어에서 측정된 프로파일링 데이터(예: nvprof, nsight 등의 결과값)를 비교
	- 최적 조합 선정
		- 비교 결과 실제 하드웨어 데이터와 가장 높은 상관관계(Correlation)를 보이는 설정을 자동으로 찾아내어, 해당 GPU 모델에 최적화된 파라미터 값으로 고정

---
### 2.2 내가 잘못 이해한 내용 & 이번에 알아가는 내용
#### 2.2.1 Trace는 HW가 바뀌어도 그대로 쓴다...?
- 특정 작업(Workload)에 대한 Trace를 한 번 생성해두면, 시뮬레이터의 설정을 변경하여 다양한 HW 구조에서의 성능을 비교/평가할 수 있는 것이 맞다.
	- 매번 Trace를 생성하는 것이 아님
	- Trace는 ISA-independent(ISA 독립적)한 중간 표현(Intermediate Representation)으로 변환되어 Performance Model로 전달됨. 
	- 즉, 프로그램의 논리적 흐름과 메모리 접근 정보는 Trace에 담겨 있고, 이 정보를 어떻게 처리(Timing 계산)할지는 Performance Model 설정에 달려있음

#### 2.2.2 그렇다고 그냥 막 HW 수정하고 Trace 그냥 그대로 쓸 수 있는 건 아님
- 하드웨어 설정을 변경할 때, 해당 변경 사항을 시뮬레이터가 인식할 수 있도록 config 파일 등을 수정하는 과정은 반드시 필요
	- 시뮬레이터(Performance Model)는 하드웨어의 구성 요소(SM 개수, L1/L2 캐시 용량, 메모리 대역폭 등)에 대한 정보를 가지고 시뮬레이션을 수행합니다. 따라서 "이전과 다른 하드웨어 구조"를 시뮬레이션하고 싶다면, 그에 맞게 시뮬레이터의 config 파일을 수정하여 시뮬레이터가 새로운 하드웨어처럼 동작하도록 만들어야 함
- Trace와 Config의 관계
	-  **Trace:** "어떤 명령어들이 어떤 순서로 실행되는가(**데이터 흐름**)"에 대한 정보. 이는 하드웨어가 바뀌어도 프로그램 자체가 바뀌지 않는 한 대부분 동일
	- **Config**: "그 명령어들을 **처리하는 하드웨어의 자원이 얼마나 있는가**"에 대한 정보
	- 결합: Accel-Sim은 고정된 Trace를 입력받고, 수정된 Config 설정에 따라 해당 명령어들이 얼마나 빨리 혹은 느리게 완료되는지를 계산

#### 2.2.3  하드웨어의 ISA 자체가 변경되는 경우(예: 기존에 없던 새로운 명령어가 추가됨)에는...? 
- Accel-Sim의 ISA Def 파일을 수정해야 함.
	- 새로운 명령어가 어떤 실행 유닛에서 처리되어야 하는지 매핑 정보를 업데이트해 주어야 함

#### 2.2.4 [2.2.1]~[2.2.3] 정리
>  HW의 설정이 변경되어도 Trace를 그대로 쓸 수 있는것 맞음.
>  그러나, Config 설정은 변경된 HW에 따라서 수정해줘야 함. 
>  그러나, 새로운 ISA가 도입되는 HW 수정이라면, ISA Def 파일까지 수정해야 함


#### 2.2.5 140개의 Workload가 있다는데, 이걸 어디다 씀...?
![[Pasted image 20260618185809.png]]
- 예를 들어, 그냥 llama3-8b의 디코드 일부분만 떼어서 돌려본다고 가정하면, 저 워크로드들 안쓰고 그냥 내가 trace 만들어서 하면되는거임
- 근데 HW 수정한 부분에 대한 성능 평가만 해보고 싶으면 쓸 수도 있는거고, 아무튼 저자들이 만들어놓은 Trace가 있다 정도로 이해하면 될 듯

---

## 3. 내가 논문에서 얻어갈 내용
### 3.1 내 연구에서 Accel-Sim을 어떻게 수정해야 하는가
#### 3.1.1 내가 제안 하는 HW가 어느정도의 수정을 요구하는지 알아야됨
- -**Config 수정만으로 끝나는 부분**: offloading 타겟이 되는 메모리 계층(L1/L2/HBM)의 용량, 레이턴시, 대역폭 파라미터 조정 등등
- **Performance Model(GPGPU-Sim 4.0) 코드 수정이 필요한 부분**: Prefetch 로직 자체 — 어떤 head를 언제 prefetch할지 결정하는 정책은 기존 구조에 없으니 C++ 코드 레벨 수정 불가피
- **ISA Def 수정이 필요한지**: offloading/prefetch를 새로운 명령어(ISA)로 추상화하지 않고 기존 LD/ST 명령어로 모델링한다면 ISA Def 수정은 불필요. 반면 prefetch instruction을 명시적으로 넣으려면 필요

### 3.2 Trace 재사용 전략
- Prefetch 정책 변경, 메모리 계층 파라미터 변경 등은 **Config 레벨에서만 바꾸면서** 반복 실험 가능
- BUT!!  Prefetch 동작 자체가 메모리 접근 패턴을 바꾼다면 — 즉, prefetch로 인해 실제로 실행되는 메모리 instruction이 달라진다면 — trace를 다시 생성해야 함
- 이 경우는 execution-driven mode로 fallback하거나, prefetch를 별도 inject하는 방식을 고려해야 함

### 3.3 Validation 인프라를 내 연구에 어떻게 활용할 것인가
- 논문이 강조하는 microbenchmark + correlator 파이프라인은 내가 HW 수정 후 "이 모델이 실제와 얼마나 가까운가"를 검증하는 데 그대로 쓸 수 있음!!
	- 기존 38개 microbenchmark 중 memory latency/bandwidth 관련 것들은 offloading 경로 검증에 바로 활용 가능
	- Prefetch 효과 측정을 위한 **custom microbenchmark를 추가**해야 할 수 있음 
		- 논문의 Figure 3 (L1 latency 벤치마크) 작성 패턴을 그대로 따르면 됨
		- ![[Pasted image 20260618191929.png]]

---

## 4. 나만의 결론
### 4.1 내 연구에서 Accel-Sim 수정의 범위와 순서 구분해야된다.
```
Layer 1 (Config만 수정) — HW 파라미터 바꾸기
  → offloading 대상 메모리 계층 설정
  → 빠름, 검증 쉬움

Layer 2 (GPGPU-Sim 4.0 C++ 수정) — Prefetch 정책 구현
  → 어떤 head를 언제 prefetch할지 스케줄링 로직
  → 시간 많이 걸림, 이 부분이 연구의 핵심 contribution

Layer 3 (ISA Def 수정) — 필요시만
  → Prefetch를 명시적 명령어로 넣으려면 필요
  → 아마 초기 단계에서는 불필요
```

### 4.2 실험 설계에서 생길 트레이드오프
- Trace는 특정 HW에서 실행된 결과물이므로, 내가 prefetch로 바꾸는 메모리 접근 패턴이 trace 자체에 반영되지 않음
- 이 문제를 어떻게 다룰지 — execution-driven mode로 전환하거나, prefetch injection을 trace에 수동으로 추가하거나 — 가 실험 설계의 핵심 결정 포인트

### 4.3 한 줄 요약
> Accel-Sim은 Trace를 고정하고 HW를 바꿔가며 실험하기에 최적화된 도구
> 
> 내 연구에서도 trace를 한 번 생성해두고 offloading/prefetch 정책과 메모리 파라미터를 바꾸며 반복 실험하는 방식으로 활용하되, 
> 
> prefetch가 instruction 흐름 자체를 바꾸는 시점에서는 execution-driven으로 전환하는 기준을 사전에 설정해두어야 함
---

## 5. 기타 정리할 내용
### 5.1 용어 정리

| 용어                      | 정의                                                                             |
| ----------------------- | ------------------------------------------------------------------------------ |
| vISA (virtual ISA)      | PTX: NVIDIA의 추상 ISA. 컴파일러가 생성, 드라이버가 mISA로 JIT 컴파일.                            |
| mISA (machine ISA)      | SASS: 특정 GPU 세대의 실제 하드웨어 명령어. sm_70=Volta, sm_75=Turing 등.                     |
| NVBit                   | NVIDIA의 dynamic binary instrumentation framework. SASS 레벨에서 동작, LD_PRELOAD 방식. |
| SASS                    | Shader ASSembly. NVIDIA GPU의 mISA.                                             |
| sub-core                | Volta 이후 SM의 기본 단위: warp scheduler + 전용 RF + 전용 EU.                            |
| sectored cache          | 128B cache line을 32B sector 4개로 분할. tag는 128B, data fetch는 32B 단위.             |
| partition camping       | $2^n$ 메모리 bank에서 $2^k$ stride 접근이 특정 bank에 집중되는 현상.                            |
| IPOLY hashing           | Galois field irreducible polynomial 기반 L2 bank index 생성. camping 완화.           |
| write-validate          | write miss 시 DRAM fetch 없이 sector에 기록 후 byte-level mask로 유효성 추적.               |
| MSHR                    | Miss Status Holding Register. in-flight cache miss 추적 구조.                      |
| simulation-gating       | 할 일 없는 컴포넌트의 cycle tick을 skip하는 속도 최적화.                                        |
| FR-FCFS                 | First-Row First-Come First-Serve. row-buffer hit를 우선 처리하는 DRAM 스케줄러.           |
| active mask             | warp 내 현재 cycle에 active한 thread 비트맵 (32bit).                                   |
| ISA Def 파일              | SASS opcode → execution unit 매핑 파일. 사용자가 공개 문서로 작성.                            |
| HW Def 파일               | SM 수, warp 크기 등 기초 HW 파라미터 정의 파일. microbenchmark가 참조.                          |
| NRMSE                   | Normalized RMSE. 참값이 0인 counter에 사용.                                           |
| Correl                  | Karl Pearson coefficient of variation (σ/μ). HW vs Sim 추세 일치도.                 |
| kernelslist             | Accel-Sim trace 디렉토리의 kernel 실행 순서 메타파일.                                       |
| base+stride compression | 규칙적 메모리 접근 패턴을 (base, stride) 쌍으로 압축.                                          |

### 5.2 논문 리딩을 위한 알아두면 좋은 영어 표현
| 영어 표현                           | 뜻                      | 사용 예시                                                                                 |
| :------------------------------ | :--------------------- | :------------------------------------------------------------------------------------ |
| **Bridge the gap**              | 격차를 줄이다/해소하다           | We introduce mechanisms to bridge the gap between industrial innovation and research. |
| **Keep up with**                | ~를 따라잡다/최신 상태를 유지하다    | Simulators must keep up with rapid changes in contemporary GPU designs.               |
| **In-flight**                   | (데이터/명령어가) 처리 중인       | GPU simulations involve thousands of in-flight memory requests.                       |
| **At the expense of**           | ~를 희생하여/대가로            | Adding detail often comes at the expense of increased simulation time.                |
| **Pinpoint**                    | 정확히 찾아내다/규명하다          | We use microbenchmarks to pinpoint latency and bandwidth changes.                     |
| **Drive (development/results)** | (결과를) 유도하다/이끌다         | These correlation results drive the development of our performance model.             |
| **Take steps to**               | ~하기 위한 조치를 취하다         | We take steps to improve the simulator's speed.                                       |
| **Pave the way for**            | ~의 토대를 마련하다            | This framework paves the way for future research in GPU systems.                      |
| **In the presence of**          | ~가 존재하는 환경에서/상황에서      | The L2 write policy conserves bandwidth in the presence of sub-sector reads.          |
| **Expose**                      | (숨겨진 것을) 드러내다/폭로하다     | Counter-by-counter validation exposes several changes in hardware.                    |
| **Prohibitively large**         | (비용/시간이) 너무 커서 실행 불가능한 | Some workloads are omitted as their trace sizes are prohibitively large.              |
| **Alleviate**                   | 완화하다                   | New design techniques alleviate the bottlenecks present in older models.              |
| **Obscured by**                 | ~에 의해 가려진/흐려진          | Some performance issues are obscured by less detailed simulation models.              |
| **In light of**                 | ~를 고려하여/비추어 볼 때        | In light of slowing Moore’s law, we must focus on domain-specific pipelines.          |
| **A trade-off between**         | ~ 사이의 상충 관계(트레이드오프)    | Simulation-gating is a trade-off between event-driven and cycle-driven simulation.    |
