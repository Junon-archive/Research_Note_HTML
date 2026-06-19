제목: HMS 논문 정리 및 HW 시뮬 구현 방향성 탐구
날짜: 2026-W25
작성자: junon
지도교수: 홍정규 교수님

---
# Bandwidth-Effective DRAM Cache for GPU s with Storage-Class Memory
## 0. 논문 한눈에 보기
- 제목: Bandwidth-Effective DRAM Cache for GPU s with Storage-Class Memory - HMS (이하 HMS)
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
## 1. 논문 핵심 사항
### 1.1 해결하고자한 핵심 문제
- GPU의 메모리 용량 문제
	- UVM, Multi-GPU를 사용해도 한계가 있음

### 1.2  핵심 아이디어
![[Pasted image 20260619011528.png]]

- **핵심 문제에 대한 해결책**: [SCM] - `DRAM의 속도와 SSD의 용량의 이점을 가진 중간 단계 HW`
	- SCM을 메모리로 사용
	- **파생되는 문제 A** :
		- 그러면 긴 쓰기 지연 시간, 높은 에너지 소비를 해결해야됨

- **문제 A 에 대한 해결책** : [DRAM Cache]
	- 이때 DRAM으로 Cache를 만들어 사용
	- **파생되는 문제 B** : 
		- 1.  DRAM Cache가 쉽게 thrashing 상태가 되어 오히려 성능을 저하시키고 대역폭을 낭비하게 됨
		- 2.  DRAM Cache 히트 여부를 확인하기 위해 매번 외부 DRAM에 태그를 조회해야 하는데, 이 과정에서 발생하는 추가적인 태그 조회 트래픽이 실제 데이터 요청과 대역폭 경쟁을 벌여 전체적인 대역폭 효율을 떨어짐

- **파생되는 문제 B-1 에 대한 해결책**: [Bypass Policy]
	- SCM-aware DRAM Cache **Bypass Policy**을 도입
	- 성능 기여도가 낮은 데이터를 DRAM에 캐싱하지 않고 바로 SCM으로 보냄

- **파생되는 문제 B-2 에 대한 해결책**: [CTC]
	- Configurable Tag Cache (CTC)
	- L2 Cache의 일부 공간을 태그 캐싱 용도로 재구성(repurpose)하여 온칩(on-chip)에서 빠르게 태그를 조회
	- **파생되는 문제 C** : 
		- CTC 미스가 발생 가능
		- 이때 태그를 조회하기 위해 여러 번의 DRAM 접근이 필요

- **파생되는 문제 C 에 대한 해결책** : [AMIL]
	- Aggregated Metadata-In-Last-column (AMIL)
	- 한 DRAM 로우(Row) 내의 모든 태그와 메타데이터를 마지막 컬럼(Last-column)에 집중적으로 배치
	- 단 한 번의 컬럼 접근으로 해당 로우의 모든 태그를 가져올 수 있게 함

### 1.3 Contribution이 뭔지
- GPU 맞춤형 S**CM-aware DRAM 캐시** 설계
	- SCM의 물리적 특성(긴 쓰기 지연 시간 및 높은 에너지)과 워크로드의 접근 패턴(공간 지역성, 접근 빈도)을 통합적으로 고려한 지능형 **바이패스 정책(SCM-aware Bypass Policy)** 을 최초로 제안
- 태그 조회 효율 극대화 
	- **Configurable Tag Cache (CTC)**
	- **Aggregated Metadata-In-Last-column (AMIL)**
- 위 사항들을 도입하여 성능 및 에너지 효율성 입증함
	- 기존의 HBM 기반 GPU 대비 성능을 최대 12.5배(평균 2.9배) 향상
	- 에너지 소비를 최대 89.3%(평균 48.1%) 절감
---

## 2. 나한테 필요한 정보
Accel-sim 어떻게 수정했는지

### 2.1 논문에서 설명한 내용
> "We integrated Accel-sim with a UM model [46] and Ramulator [77] for simulation." 

관련 설명 위 한 문장이 다임... 코드를 직접 봐야 알 수 있음

### 2.2 코드 구조
#### 2.2.1 레포 구조 파악
핵심 발견: 3-repo 구조

```
/home/junon/HMS/
├── HMS/accelsim_HMS/          ← trace front-end (PSAL fork)
│   └── gpu-simulator/
│       ├── main.cc            ← cudaMalloc/page init, GMMU 연동
│       ├── trace-driven/      ← trace_driven.cc/h
│       ├── trace-parser/      ← trace_parser.cc/h
│       └── ISA_Def/           ← opcode 정의
│
├── Accel-sim/accel-sim-framework/   ← 원본 (비교용)
│
└── gpgpu-sim_HMS/              ← ★ 실제 HMS 구현체 (빌드 시 자동 clone)
    └── src/
        ├── gpgpu-sim/
        │   ├── l2cache.cc/h   ← L2 → DRAM 인터페이스
        │   ├── gmmu.cc/h      ← 페이지 폴트, PCIe, 프리페치
        │   └── gpu-sim.cc/h   ← 글로벌 설정
        └── ramulator/
            ├── HMSController.cc/h  ← ★ DRAM cache 핵심 로직
            ├── HMSRequest.cc/h     ← request 생성
            ├── HMS.cc/h            ← 타이밍 파라미터
            ├── PCM.cc/h            ← SCM 모델
            └── HBM.cc/h            ← HBM 모델
```

> `gpgpu-sim_HMS`는 `setup_environment.sh`가 빌드 시 자동으로 clone한다 ([setup_environment.sh](vscode-webview://1hsmnbochkh6bhu021dbgd8c0fttlao6o38jqhif1rjm7k2veqvm/HMS/accelsim_HMS/gpu-simulator/setup_environment.sh)). 원본 대비 accelsim_HMS가 추가/수정한 내용은 주로 Jenkins CI 파일, `generate_pagetable.ipynb`, trace-driven 레이어이며 DRAM cache 로직은 전부 gpgpu-sim_HMS에 있다.

이 두 파일이 핵심
```
ramulator/HMSController.cc  ← DRAM cache 전체 로직
gpgpu-sim/gmmu.cc           ← page fault, PCIe, prefetch
```

#### 2.2.2 accelsim_HMS vs 원본 주요 차이 (파일 수준)

| 변경 유형          | 주요 내용                                                                        |
| -------------- | ---------------------------------------------------------------------------- |
| **추가**         | `generate_pagetable.ipynb`, `.travis.yml`, Jenkinsfile 3개                    |
| **삭제**         | `util/` 전체, `python_wrapper/`, CMakeLists.txt 전체, GitHub Actions             |
| **수정**         | `main.cc`, `trace_driven.cc/h`, `trace_parser.cc/h`, ISA_Def 헤더 6개, 설정 파일 4개 |
| **제거된 GPU 설정** | SM7_GV100, SM80_A100, SM86_RTX3070                                           |

### 2.3 내가 잘못 이해한 내용
- HMS 레포 자체에 DRAM cache 코드가 있을 거라고 생각했는데
	- 실제로는 `setup_environment.sh`가 실행 시 `gpgpu-sim_HMS`를 별도로 clone해서 `/tmp/`에 까는 구조였음. 
	- -> 즉 진짜 수정 코드는 레포 밖에 있음
- AMIL, RHBM 같은 논문 용어가 코드에도 그대로 있을 거라 생각했는데, 실제로는 기능적 등가물이 다른 이름으로 구현되어 있음

### 2.5 이번에 알아가는 내용
- 진짜 수정 코드는 전부 `gpgpu-sim_HMS/src/` 안에 있고
- 핵심 파일은 두 개로 압축됨
    - `ramulator/HMSController.cc` : DRAM cache 전체 로직 (bypass policy, tag cache, fill)
    - `gpgpu-sim/gmmu.cc` : page fault, PCIe 지연, prefetch
- 모든 memory request는 `HMSController::enqueue()`를 첫 관문으로 통과함. 
	- 여기서 주소 기반으로 DRAM/SCM/PCIe 경로를 분기함
- HMS 전체를 관통하는 하나의 패턴이 있음: **"판단 → 큐잉 → tick에서 실행"**. 
	- bypass policy도, prefetch도 전부 이 패턴으로 동작함
---

## 3. 내가 논문에서 얻어갈 내용
- **"판단 → 큐잉 → tick 실행" 패턴** : 내 Markov prefetch scheduler도 이 패턴으로 구현하면 됨. 
	- `post_process_cache_miss() → fill_pending queue → tick()` 흐름을 그대로 재활용
	- 판단: 요청이 들어온 시점에, 이 요청을 어떻게 처리할지 결정하는 단계
	- 큐잉: **큐에 넣는** 단계
	- 실행: tick() 함수가 큐에서 요청을 꺼내서 실제로 처리하는 단계

- HMS 논문 코드에서 배운 패턴

```
실제 GPU 요청
    ↓
HMSController::enqueue()   ← 첫 관문, 경로 판단
    ↓
[DRAM / SCM / PCIe] 분기
    ↓
tick()에서 실행            ← "판단 → 큐잉 → tick 실행" 패턴
```

내 상황에서는 아래처럼, 패턴 자체는 동일하게 재사용 가능

```
Attention 계산 요청
    ↓
HVRPController::enqueue()  ← head_id 판별 + SRAM/HBM 분기
    ↓
[SRAM(on-chip table) / HBM(KV cache)] 분기
    ↓
tick()에서 prefetch 실행
```

- **PCIE_VALIDATE 큐** : 
	- head를 fetch할 경로가 이미 구현되어 있음 (확인 필요)
- **Trace 재사용 전략** : 
	- HMS도 SASS trace 그대로 쓰고 page 메타데이터만 별도로 붙이는 방식. 내 연구도 주소 → head_id 매핑 파일만 따로 만들면 trace 재생성 없이 가능
---

## 4. 나만의 결론
### 4.1 수정 방향성
- 구조 자체는 Accel-sim/GPGPU-sim 그대로 재활용하고, 필요한 추가로 주입하는 방식으로 접근하면 됨.

수정 범위 요약:

**Layer 1 — Config 수정만으로 끝나는 부분**
- Predictor SRAM 용량·레이턴시 파라미터 (새 캐시를 기존 L1/L2 계층과 다른 레이턴시로 설정)
- HBM 대역폭 파라미터 (KV 접근 baseline 설정)
- 이건 `.config` 파일 수정으로 처리 가능

**Layer 2 — GPGPU-Sim C++ 코드 수정 (핵심 기여)**
- HMS의 `HMSController.cc`에 대응되는 게 내 연구에서는 두 파일:

1. **Predictor SRAM Controller** (신규 작성): Markov 테이블 조회 로직. `enqueue()` 받아서 head_id → table hit/miss 판별 → tick에서 결과 반환.
2. **KV Prefetch Scheduler** (gmmu.cc 패턴 참고): 테이블 히트 시 다음 토큰의 KV를 미리 HBM에서 fetch. HMS의 `PCIE_VALIDATE` 큐 패턴을 여기서 재활용할 수 있어 — CPU-resident 데이터 fetch 경로가 이미 구현되어 있으니, 분기 조건만 head_id 기반으로 추가하면 됨.

**Layer 3 — ISA Def 수정**
- 나의 경우는 기존 LD/ST로 KV 접근을 모델링하는 한 **초기 단계에서 불필요**. 
- Prefetch를 memory controller가 투명하게 처리하면 기존 SASS trace 그대로 사용 가능.

### 4.2 [핵심] 구현의 방향성을 잡아보자
구현할 내용 크게 세 개

| **대상**                 | **평가 내용**                                           |
| ---------------------- | --------------------------------------------------- |
| Predictor cache        | 용량, lookup latency, hit rate, DRAM fallback, 면적, 전력 |
| KV prefetch controller | bandwidth 활용률, latency hiding, wasted prefetch      |
| Selective KV load      | memory traffic 감소량, decode latency 변화               |

#### 4.2.1 Head마다 달라지는 kv 이동을 시뮬레이터에서 어떻게 표현?
- ISA 수정 없이, 메모리 컨트롤러 레벨에서 mask로 처리하는 방식 생각중

**구체적으로??**:
- KV cache에서 선택된 8개(K=8일 때 가정) head의 데이터에만 유효성 비트를 세우고, 나머지 8개 head의 fetch 요청을 컨트롤러 레벨에서 suppress
- 이렇게 하면 SASS trace에 있는 기존 LD 명령어를 그대로 쓰면서, memory controller가 실제로 HBM에 나가는 트래픽만 줄이는 효과를 만들 수 있지 않을까 싶음

#### 4.2.2 Predictor SRAM을 시뮬레이터에서 어떻게 모델링할 거냐?
- 기존 L1 캐시 구조를 별도로 하나 더 만들고
- 용량·레이턴시·포트 수만 SRAM 스펙에 맞게 Config에서 조정하는 방식 생각중

**구체적으로??**:
- Accel-sim의 캐시 구조는 용량, associativity, hit/miss latency를 config 파라미터로 받음
	- 여기에 "Predictor SRAM"이라는 이름의 새 캐시 인스턴스를 추가
- 레이턴시는 SRAM 면적/전력을 먼저 추정하고, 거기서 나온 access latency를 config에 주입한다. (어케 추정함??) (확인 필요)
- 테이블 lookup 요청이 이 SRAM 인스턴스에 먼저 들어오고, miss 시에만 DRAM(실제로는 발생 안 한다고 가정 — 테이블이 SRAM에 완전히 올라간다고 가정)으로 fallback하는 경로를 만들기

####  4.2.3  Markov 테이블 조회와 KV Prefetch를 어떻게 파이프라인으로 연결할 거냐?
조회 결과가 나오기 전에 prefetch를 언제 트리거하냐, 레이턴시 겹침을 어떻게 모델링할 것인가?

- HMS의 "판단 → 큐잉 → tick 실행" 패턴을 그대로 써도 될 듯
- 조회 요청을 큐에 넣는 순간 prefetch 큐도 동시에 세우는 방식 생각중

**구체적으로>??**
- 토큰 t의 attention 시작 시점에 SRAM 조회 요청을 enqueue
- SRAM hit이 확인되는 순간(hit latency 후) KV prefetch 큐에 해당 head들의 HBM 주소를 넣기
- miss 시에는 원본 라우터 fallback 경로 — 이건 기존 HBM 접근 경로 그대로

---

## 5. 기타 정리할 내용
### 5.1 용어 정리
| 용어                  | 의미 (Description)                                                   | 관련 기술       |
| ------------------- | ------------------------------------------------------------------ | ----------- |
| SCM                 | Storage-Class Memory: DRAM과 Flash 사이의 특성을 가진 비휘발성 메모리. 예: PCM      | 시스템 기반      |
| HMS                 | Heterogeneous Memory Stack: DRAM과 SCM을 3D 적층 구조로 결합한 메모리 조직        | 시스템 구조      |
| TAD                 | Tag-And-Data: 태그와 데이터를 함께 저장하는 전통적인 DRAM 캐시 조직 방식                  | AMIL 비교군    |
| CTC                 | Configurable Tag Cache: L2 캐시의 일부를 DRAM 캐시 태그 저장소로 전환하는 기법         | 태그 조회 최적화   |
| AMIL                | Aggregated Metadata-In-Last-column: 로우 내 마지막 컬럼에 메타데이터를 집중 배치하는 구조 | 태그 조회 최적화   |
| Row Buffer Locality | 동일 로우 내의 데이터를 연속적으로 접근하여 성능을 높이는 척도                                | SCM penalty |
| UM                  | Unified Memory: CPU/GPU가 단일 주소 공간을 공유하며 자동 데이터 이동을 수행하는 방식         | 배경 지식       |
| MSHR                | Miss Status Holding Register: 캐시 미스 발생 시 보류 중인 요청을 관리하는 하드웨어 구조    | 구현 세부       |

### 5.2 논문 리딩을 위한 알아두면 좋은 영어 표현
| 영어 표현                   | 뜻                               | 사용 예시 (본문 변형)                                                     |
| ----------------------- | ------------------------------- | ----------------------------------------------------------------- |
| Mandate                 | ~을 필수로 요구하다                     | “Workloads that mandate memory oversubscription.”                 |
| Capture                 | 데이터를 수용하다 / 보관하다 / 포함하다         | “The GPU can capture a larger fraction of the memory footprint.”  |
| Repurpose               | 재사용하다 / 용도를 바꾸다                 | “Repurposing part of the L2 cache to cache DRAM tags.”            |
| Curtail                 | 축소하다 / 억제하다                     | “SCM throttling to curtail power consumption.”                    |
| Amortize                | 시간이나 비용을 여러 접근에 나누어 상쇄하다        | “Amortize the long activation latency of SCM.”                    |
| Orthogonal to           | ~와 독립적인 / 상관없는                  | “These approaches are orthogonal to our design.”                  |
| Benefit from            | ~로부터 이점을 얻다 / 혜택을 받다            | “Workloads with high BW demand benefit more from CTC.”            |
| Be pronounced           | 현상이나 효과가 두드러지다                  | “The speedup was especially pronounced for graph workloads.”      |
| Account for             | ~의 비율을 차지하다 / ~에 해당하다           | “Last column accounts for a very small fraction of data.”         |
| Sublinear / Superlinear | 선형 미만 / 선형 초과로 증가하는             | “They superlinearly increase system cost.”                        |
| Mitigate                | 위험이나 영향을 완화하다                   | “Effectively mitigate the thermal issue.”                         |