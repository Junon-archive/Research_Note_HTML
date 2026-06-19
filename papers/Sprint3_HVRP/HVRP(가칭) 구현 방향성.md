제목: HVRP 구현 방향성
날짜: 2026-W25
작성자: junon
지도교수: 홍정규 교수님

---
# HVRP(가칭)
## 1. 내가 HMS 논문에서 얻어갈 내용
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

## 2. 나만의 결론
### 2.1 수정 방향성
- 구조 자체는 Accel-sim 그대로 재활용하고, 필요한 추가로 주입하는 방식으로 접근하면 됨.

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

---
## 3. 나만의 결론
### 3.1 수정 방향성
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

### 3.2 [핵심] 구현의 방향성을 잡아보자
구현할 내용 크게 세 개

| **대상**                 | **평가 내용**                                           |
| ---------------------- | --------------------------------------------------- |
| Predictor cache        | 용량, lookup latency, hit rate, DRAM fallback, 면적, 전력 |
| KV prefetch controller | bandwidth 활용률, latency hiding, wasted prefetch      |
| Selective KV load      | memory traffic 감소량, decode latency 변화               |

#### 3.2.1 Head마다 달라지는 kv 이동을 시뮬레이터에서 어떻게 표현?
- ISA 수정 없이, 메모리 컨트롤러 레벨에서 mask로 처리하는 방식 생각중

**구체적으로??**:
- KV cache에서 선택된 8개(K=8일 때 가정) head의 데이터에만 유효성 비트를 세우고, 나머지 8개 head의 fetch 요청을 컨트롤러 레벨에서 suppress
- 이렇게 하면 SASS trace에 있는 기존 LD 명령어를 그대로 쓰면서, memory controller가 실제로 HBM에 나가는 트래픽만 줄이는 효과를 만들 수 있지 않을까 싶음

#### 3.2.2 Predictor SRAM을 시뮬레이터에서 어떻게 모델링할 거냐?
- 기존 L1 캐시 구조를 별도로 하나 더 만들고
- 용량·레이턴시·포트 수만 SRAM 스펙에 맞게 Config에서 조정하는 방식 생각중

**구체적으로??**:
- Accel-sim의 캐시 구조는 용량, associativity, hit/miss latency를 config 파라미터로 받음
	- 여기에 "Predictor SRAM"이라는 이름의 새 캐시 인스턴스를 추가
- 레이턴시는 SRAM 면적/전력을 먼저 추정하고, 거기서 나온 access latency를 config에 주입한다. (어케 추정함??) (확인 필요)
- 테이블 lookup 요청이 이 SRAM 인스턴스에 먼저 들어오고, miss 시에만 DRAM(실제로는 발생 안 한다고 가정 — 테이블이 SRAM에 완전히 올라간다고 가정)으로 fallback하는 경로를 만들기

####  3.2.3  Markov 테이블 조회와 KV Prefetch를 어떻게 파이프라인으로 연결할 거냐?
조회 결과가 나오기 전에 prefetch를 언제 트리거하냐, 레이턴시 겹침을 어떻게 모델링할 것인가?

- HMS의 "판단 → 큐잉 → tick 실행" 패턴을 그대로 써도 될 듯
- 조회 요청을 큐에 넣는 순간 prefetch 큐도 동시에 세우는 방식 생각중

**구체적으로>??**
- 토큰 t의 attention 시작 시점에 SRAM 조회 요청을 enqueue
- SRAM hit이 확인되는 순간(hit latency 후) KV prefetch 큐에 해당 head들의 HBM 주소를 넣기
- miss 시에는 원본 라우터 fallback 경로 — 이건 기존 HBM 접근 경로 그대로

---
## 4. 구현 위치
### 4.1 SRAM Cache
- L2와 HBM 사이, 전용 on-chip SRAM
	- L1/Shared Memory는 SM마다 독립적이라 테이블 공유에 부적합
	- L2는 용량이 부족
	- 그렇다고 HBM에만 두면 SW lookup 문제

```
SM들
 │
 ├── L1/Shared Memory (기존, 건드리지 않음)
 │
 L2 Cache (기존, 건드리지 않음)
 │    │
 │    └── [Predictor SRAM] ← ★ 여기
 │         - 테이블 일부 상주
 │         - 모든 SM에서 shared 접근
 │         - 읽기 전용
 │
HBM (나머지 테이블 + KV cache)
```

### 4.2 UVM
- GPGPU-sim 자체에서 지원하지 않으나 간단한 수정으로 구현이 충분히 가능함
	- HMS도 그렇고 다른 사람들도 그렇고 각자 알아서 구현해서 쓰는 중
	- 자세한 구현 방법 타 논문 구현한 내용 확인해보고 구현해보는게 좋겠음

### 4.3 Prefetch Scheduler
![[Pasted image 20260619113237.png]]
- 토큰 t에서 Markov 테이블 조회 결과가 나오면, 
- **토큰 t+1의 attention에서 필요할 KV cache** — 선택된 k개 head의 K, V — 를 미리 HBM에서 가져오는 것
```
토큰 t 처리 중
    ↓
SRAM에서 테이블 조회 → "다음에 head {2,5,7,9,11,13,14,15} 쓸 것 같다"
    ↓
이 8개 head의 KV를 HBM에서 미리 fetch
    ↓
토큰 t+1 attention 시작할 때 KV가 이미 올라와 있음
```

- HMS의 gmmu.cc 파일 수정한 내용을 참조
	- page fault 기반이 아니라 **prediction 기반 prefetch**라서 위치는 다름
```
SM (Attention 연산)
 │
 │  Predictor SRAM 조회 결과
 │         ↓
L2 Cache ──┤
 │         └── [KV Prefetch Scheduler] ← ★ 여기
 │               - L2와 HBM 사이
 │               - 조회 결과 받아서 HBM fetch 요청 생성
 │               - 기존 demand fetch와 우선순위 조정
 │
HBM (KV cache 상주)
```

- prefetch 완료된 KV가 L2를 거쳐 SM에 올라오는 경로가 기존 demand fetch와 동일해서 추가 경로 불필요
- HMS의 HMSController::enqueue()가 정확히 이 위치에서 경로를 분기한 것과 동일한 패턴
- SM 레벨에 두면 SM마다 독립 스케줄러가 생겨서 중복 fetch 발생 가능

---
## 5.0 Accel-sim에서 구체적으로 어느 파일을 수정하나
### 5.1 HMS 코드 구조를 참고
```'
gpgpu-sim/
├── l2cache.cc       ← L2 → DRAM 인터페이스, 여기서 prefetch 요청 inject
├── gmmu.cc          ← HMS가 prefetch 큐 관리한 파일, 패턴 참고
└── gpu-sim.cc       ← 전역 설정, Scheduler 인스턴스 등록
```

### 5.2 내가 수정해야할 파일
```
gpgpu-sim/
└── hvrp_prefetch.cc  ← Prefetch Scheduler 본체
    hvrp_prefetch.h
```
---
## 6.0 고려해봐야하는 문제점
### 6.1 prefetch 타이밍을 어떻게 맞출 거냐
- 너무 일찍 하면 L2에서 evict되고 너무 늦으면 무의미
	- 일찍 fetch해서 L2에서 evict되는 비율과 늦어서 demand와 겹치는 비율을 stat으로 뽑아서 최적 window를 찾는 실험 필요할 것으로 예상

### 6.2 Prefetch가 demand fetch랑 bandwidth 경쟁하면 오히려 느려지지 않냐
- HMS의 Bypass Policy처럼, Fetch의 정책을 정확히 수립해놔야함
	- Prefetch가 항상 우선이고, demand fetch는 남은 bandwidth만 쓰는 구조

### 6.3 SRAM Cache 구성 시 tag 확인하는 과정이 필요한가? 
- 이로인해 생기는 오버헤드도 있을까
- Claude 왈
	- 네 SRAM의 특수성
	- 네 Predictor SRAM은 일반 캐시와 결정적으로 다른 점이 있어:
	- **테이블의 key가 곧 주소야.**

```
조회 요청: (layer_idx, prev_8set, combo5)
               ↓
        uint32 packed key 생성
               ↓
        이 key로 SRAM 내 위치를 직접 계산 가능
```

- 일반 캐시처럼 "임의의 메모리 주소가 어느 캐시 라인에 매핑되는지" 모르는 상황이 아니야. key 자체가 구조화되어 있고, 테이블이 key 순으로 정렬되어 있어.
