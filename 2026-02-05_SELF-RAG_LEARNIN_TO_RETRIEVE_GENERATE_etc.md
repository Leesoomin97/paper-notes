## 1. Introduction

### 1.1 연구 배경

최근 대규모 언어 모델(LLM)은 모델 크기와 학습 데이터 규모가 크게 증가했음에도 불구하고, 여전히 사실 오류(factual error) 문제를 완전히 해결하지 못하고 있음. 이는 지식 집약적 질의응답이나 장문 생성 과제에서 특히 두드러지게 나타남.

이러한 문제를 완화하기 위한 방법으로 **Retrieval-Augmented Generation (RAG)**이 제안되었으며, 외부 문서를 검색해 입력에 추가함으로써 모델의 사실 정확도를 향상시켜 왔음.

---

### 1.2 기존 RAG의 한계

기존 RAG 방식은 다음과 같은 한계를 가짐.

- 검색 필요 여부와 관계없이 **항상 문서를 검색**함
- 불필요하거나 질문과 직접적인 관련이 없는 문서가 포함될 수 있음
- 모델이 검색 문서를 따르도록 명시적으로 학습되지 않아  
  **검색 문서와 생성 결과 간의 불일치**가 발생함

이로 인해 RAG가 LLM의 범용성과 생성 품질을 저해하는 경우도 존재함.

---

### 1.3 SELF-RAG의 제안

본 논문은 이러한 문제를 해결하기 위해  
**Self-Reflective Retrieval-Augmented Generation (SELF-RAG)**을 제안함.

SELF-RAG의 핵심 아이디어는 모델이 단순히 답변을 생성하는 데 그치지 않고,  
**스스로 검색이 필요한지 판단하고 생성 결과의 품질을 평가하도록 만드는 것**임.

---

### 1.4 SELF-RAG의 핵심 구성

SELF-RAG는 생성 과정에서 일반 텍스트와 함께  
**reflection token**을 출력함.

- **retrieval token**  
  현재 생성 과정에서 외부 검색이 필요한지를 판단
- **critique token**  
  생성된 답변의 사실성 및 전반적인 품질을 평가

---

### 1.5 생성 흐름

![image.png](attachment:73b6cf9d-bae9-4d5d-ba4b-b5d05b5f736a:image.png)

SELF-RAG의 생성 과정은 다음과 같음.

1. 입력과 이전 생성 결과를 바탕으로 검색 필요 여부를 판단
2. 필요할 경우 여러 검색 문서를 활용해 답변을 생성
3. 생성된 결과를 스스로 평가하여 가장 적절한 출력을 선택

이 과정은 항상 고정된 수의 문서를 검색하고 생성 결과를 재검토하지 않는  
기존 RAG와 차별화됨.

---

### 1.6 성능 요약

실험 결과, SELF-RAG는 기존 LLM 및 일반적인 RAG 방식 대비  
전반적인 생성 품질과 사실성, 특히 **인용 정확도 측면에서 우수한 성능**을 보임.

---

## 2. Related Work

### 2.1 Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation(RAG)은 언어 모델 입력에 외부 문서를 검색해 추가하는 방식으로, 지식 집약적 태스크에서 성능을 크게 향상시켜 왔음.

최근 연구에서는:

- 검색 문서를 고정 개수로 입력에 포함
- retriever와 LM 공동 학습
- 태스크 데이터 기반 few-shot fine-tuning

등이 제안되었으나, 대부분 **생성 시작 시점에 한 번만 검색**하는 구조를 가짐.

이를 개선하기 위해 생성 중 적응적 검색이나 API 호출 기반 접근이 제안되었으나,

- 런타임 효율 저하
- 불필요한 문맥 취약성
- 명확한 근거(attribution) 부족

등의 문제가 동반됨.

SELF-RAG는 이러한 한계를 해결하기 위해  
**on-demand retrieval + self-reflection** 구조를 제안함.

---

### 2.2 Concurrent RAG 연구

동시대 연구들은 다음과 같은 접근을 사용함.

- retriever–LM 단계적 fine-tuning
- 외부 NLI / 요약 모델을 통한 문서 필터링
- 트리 탐색 기반 생성(LATS 등)

SELF-RAG는

- 검색 필요 판단을 **self-reflection**으로 수행
- 외부 모델 의존 없이 문서 병렬 처리
- 세분화된 reflection token으로 정밀 제어

를 가능하게 한다는 점에서 차별화됨.

---

### 2.3 Critic 기반 학습 및 제어 토큰 연구

기존 RLHF는 비용이 크고 복잡함.  
SELF-RAG는 RLHF 대신 **critic 모델이 생성한 reflection token을 활용한 지도학습**을 사용함.

이를 통해:

- 낮은 학습 비용
- fine-grained 평가
- 검색 + 생성 동시 제어

가 가능해짐.

---

## 3. Self-RAG: Learning to Retrieve, Generate, Critique

SELF-RAG는 검색과 자기 반성을 결합한 프레임워크로,  
필요할 때만 검색을 수행하고 생성 결과를 스스로 평가하도록 설계됨.

모델은 일반 텍스트와 함께 **reflection token**을 출력하도록 end-to-end 학습됨.

---

### 3.1 문제 정의 및 전체 구조

![image.png](attachment:8cb06b0c-ce31-4029-982a-5ddeefee6f1e:image.png)

입력 `x`가 주어졌을 때, 모델 `M`은 출력 세그먼트  
`y = [y₁, …, y_T]`를 순차적으로 생성함.

각 세그먼트는 텍스트 토큰과 reflection token을 포함할 수 있음.

추론 시:

- 검색 필요 여부 판단
- 검색 수행 후 문서별 관련성·지원성 평가
- 전체 유용성 평가 후 최종 출력 선택

을 수행함.

---

### 3.2 SELF-RAG 학습 방식

SELF-RAG는 두 모델을 학습함.

- **Critic 모델 (C)**
- **Generator 모델 (M)**

#### 3.2.1 Critic 모델 학습

- GPT-4로 reflection token supervision 데이터 생성
- 해당 데이터를 distillation하여 critic 모델 학습

손실 함수: L_critic = E_{(x, c, y, r) ~ D_critic} [ -log P(r | x, c, y) ]

목표는 GPT-4 수준의 판단 능력을 소형 모델이 재현하도록 하는 것임.

---

#### 3.2.2 Generator 모델 학습

- critic + retriever를 사용해 데이터 보강
- reflection token을 일반 토큰처럼 포함
- next-token prediction 방식으로 학습

불필요한 검색 남용을 방지하기 위해  
검색이 필요 없는 데이터도 함께 학습함.

---

### 3.3 SELF-RAG 추론 단계 제어

#### 1) 적응형 검색
- P([Retrieve]=Yes | x, y_prev) > τ

조건 만족 시에만 검색 수행.

#### 2) 트리 디코딩

![image.png](attachment:9ff16b19-a3ab-4e23-8f49-855d91c2ac46:image.png)

- 문서별 병렬 후보 생성
- reflection token 기반 Beam Search 유사 구조

#### 3) 랭킹 수식
- Score(y, d) = P(IsREL) + w1 · P(IsSUP) + w2 · P(IsUSE)

---

## 5. 실험 분석

### 5.1 벤치마크

- Closed-set QA (ARC, OpenBookQA)
- Short-form QA (PopQA, TriviaQA)
- Long-form QA (ASQA)
- Fact Checking (FactScore)

### 5.2 결과 요약

- Self-RAG 7B가 기존 LLM 및 일반 RAG 대비 전반적 우수
- FactScore 기준 약 **10%p 이상 사실성 향상**
- 적응형 검색으로 검색 비용 최적화

---

## Appendix

### A. Reflection Token 정의

[Retrieve] : Yes / No / Continue
[IsREL] : Relevant / Irrelevant
[IsSUP] : Fully / Partially / No support
[IsUSE] : -1 ~ 5

### B. 학습 하이퍼파라미터

Base Model : Llama-2-7B / 13B
Learning Rate : 2e-5
Batch Size : 128
Context Length: 4096
Epochs : 3

### C. Critic 프롬프트 예시

질문 x와 문서 d가 주어졌을 때,
문서가 질문에 직접적인 정보를 제공하면 Relevant,
그렇지 않으면 Irrelevant를 출력하라.

### D. 가중치 설정 예시

- 사실성 중심: `w_IsSUP = 1.0`
- 유용성 중심: `w_IsUSE = 0.8`

