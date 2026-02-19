## 1. Introduction

### 1.1 연구 배경

최근 대규모 언어 모델(LLM)은 모델 크기와 학습 데이터 규모가 크게 증가했음에도 불구하고, 여전히 사실 오류(factual error) 문제를 완전히 해결하지 못하고 있음. 이는 지식 집약적 질의응답이나 장문 생성 과제에서 특히 두드러지게 나타남.

이러한 문제를 완화하기 위한 방법으로 **Retrieval-Augmented Generation (RAG)**이 제안되었으며, 외부 문서를 검색해 입력에 추가함으로써 모델의 사실 정확도를 향상시켜 왔음.

### 1.2 기존 RAG의 한계

기존 RAG 방식은 다음과 같은 한계를 가짐.

- 검색 필요 여부와 관계없이 **항상 문서를 검색**함
- 불필요하거나 질문과 직접적인 관련이 없는 문서가 포함될 수 있음
- 모델이 검색 문서를 따르도록 명시적으로 학습되지 않아, **검색 문서와 생성 결과 간의 불일치**가 발생함

이로 인해 RAG가 LLM의 범용성과 생성 품질을 저해하는 경우도 존재함.

### 1.3 SELF-RAG의 제안

본 논문은 이러한 문제를 해결하기 위해 **Self-Reflective Retrieval-Augmented Generation (SELF-RAG)**을 제안함.

SELF-RAG의 핵심 아이디어는 모델이 단순히 답변을 생성하는 데 그치지 않고, **스스로 검색이 필요한지 판단하고 생성 결과의 품질을 평가하도록 만드는 것**임.

### 1.4 SELF-RAG의 핵심 구성

SELF-RAG는 생성 과정에서 일반 텍스트와 함께 **reflection token**을 출력함.

- **retrieval token**: 현재 생성 과정에서 외부 검색이 필요한지를 판단
- **critique token**: 생성된 답변의 사실성 및 전반적인 품질을 평가

### 1.5 생성 흐름

![image.png](attachment:73b6cf9d-bae9-4d5d-ba4b-b5d05b5f736a:image.png)

SELF-RAG의 생성 과정은 다음과 같음.

1. 입력과 이전 생성 결과를 바탕으로 검색 필요 여부를 판단함
2. 필요할 경우 여러 검색 문서를 활용해 답변을 생성함
3. 생성된 결과를 스스로 평가하여 가장 적절한 출력을 선택함

이 과정은 항상 고정된 수의 문서를 검색하고 생성 결과를 재검토하지 않는 기존 RAG와 차별화됨.

### 1.6 성능 요약

실험 결과, SELF-RAG는 기존 LLM 및 일반적인 RAG 방식 대비 전반적인 생성 품질과 사실성, 특히 **인용 정확도 측면에서 우수한 성능**을 보임.

---

## 2. Related work

### 2.1 Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation(RAG)은 언어 모델의 입력에 외부 문서를 검색해 추가하는 방식으로, 지식 집약적 태스크에서 성능을 크게 향상시켜 왔음. 기존 연구들은 RAG를 통해 사전학습된 LLM이나 fine-tuning된 모델 모두에서 사실 정확도가 개선됨을 보였음.

최근 연구에서는 검색 문서를 입력 앞에 고정된 개수로 붙여 instruction-tuning을 수행하거나, retriever와 LM을 공동으로 학습한 뒤 태스크 데이터에 대해 few-shot fine-tuning을 수행하는 방식이 제안됨. 다만 이러한 접근들은 대부분 생성 시작 시점에 **한 번만 검색을 수행**하는 구조를 가짐.

이를 개선하기 위해, 일부 연구는 생성 과정 중 **적응적으로 검색**을 수행하거나, 특정 엔티티에 대해 API 호출을 생성하도록 모델을 학습함. 그러나 이러한 방법들은 태스크 성능 향상과 동시에

- 런타임 효율 저하,
- 불필요한 문맥에 대한 취약성,
- 출력 결과에 대한 명확한 근거(attribution) 부족

등의 문제를 동반하는 경우가 많음.

본 논문은 이러한 한계를 해결하기 위해, 임의의 LM이 **필요할 때만 검색을 수행(on-demand retrieval)**하도록 학습하는 방법을 제안함. 또한 reflection token을 활용해 생성 과정을 제어함으로써, 생성 품질과 인용 정확도를 동시에 개선하고자 함.

### 2.2 Concurrent RAG 연구

동시대의 RAG 연구들 역시 기존 RAG 구조를 개선하기 위한 다양한 학습 및 프롬프팅 전략을 제안함.

일부 연구는 retriever와 LM을 단계적으로 instruction-tuning 데이터셋에 맞게 fine-tuning하는 방식을 사용함. 이와 달리 SELF-RAG는 다양한 instruction-following 데이터셋으로 학습되면서도, **검색 필요 여부 판단과 출력 선택을 self-reflection으로 수행**한다는 점에서 더 범용적이고 제어 가능한 구조를 가짐.

다른 연구들은 검색된 문서를 필터링하거나 요약하기 위해 자연어 추론 모델이나 요약 모델과 같은 **외부 모델**을 활용함. 반면 SELF-RAG는 검색 문서를 병렬적으로 처리하고, self-reflection을 통해 불필요한 문서를 제거함으로써 **추론 시 외부 모델에 의존하지 않음**. 또한 SELF-RAG의 self-reflection은 문서 관련성뿐 아니라 사실성 등 출력 품질의 다양한 측면을 평가함.

LATS와 같은 접근법은 off-the-shelf LLM을 사용해 검색과 트리 탐색 기반 생성을 수행하며, LM이 생성한 점수로 출력을 평가함. 그러나 이러한 방식은 출력 전체에 대한 단일 점수만을 제공하는 반면, SELF-RAG는 **세분화된 reflection token을 생성하도록 학습**하여 보다 정밀한 평가와 커스터마이즈 가능한 추론을 가능하게 함.

### 2.3 Critic 기반 학습 및 제어 토큰 관련 연구

인간 피드백 기반 강화학습(RLHF)은 LLM을 인간 선호에 맞게 정렬하는 데 효과적인 방법으로 알려져 있음. 최근에는 여러 reward model을 활용한 fine-grained RLHF도 제안됨.

SELF-RAG 역시 생성 결과에 대한 세분화된 평가를 다루지만, RLHF 대신 **critic 모델이 오프라인으로 생성한 reflection token을 활용해 지도 학습**을 수행함. 이를 통해 RLHF 대비 훨씬 낮은 학습 비용으로 모델을 학습할 수 있음.

또한 기존의 control token 기반 연구들은 생성 방향을 유도하는 데 초점을 두는 반면, SELF-RAG는 reflection token을 통해 **검색 필요 판단과 생성 결과 평가를 동시에 수행**한다는 점에서 차별화됨. 일부 self-evaluation 기반 디코딩 연구는 추론 태스크에만 초점을 맞추고 검색을 포함하지 않지만, SELF-RAG는 retrieval과 생성 전반을 아우르는 구조를 가짐.

마지막으로, 출력 생성 → 피드백 → 재생성을 반복하는 LLM refinement 계열 연구들도 존재하나, 이는 추론 비용이 크게 증가한다는 한계가 있음.

---

## 3. Self-RAG: Learning to Retrieve, Generate, Critique

SELF-RAG(Self-Reflective Retrieval-Augmented Generation)는 검색과 자기 반성(self-reflection)을 결합하여, LLM의 사실성과 생성 품질을 향상시키면서도 기존 LLM의 창의성과 범용성을 유지하는 프레임워크임. 기존 RAG와 달리, 필요할 때만 검색을 수행하고 생성 결과를 스스로 평가하도록 설계됨.

SELF-RAG는 end-to-end 학습을 통해, 모델이 검색 문서를 참고한 텍스트 생성과 함께 **특수한 reflection token**을 출력하도록 학습됨. 이 토큰들은 검색 필요 여부를 나타내거나, 생성 결과가 검색 문서에 의해 충분히 뒷받침되는지를 평가하는 역할을 함. 기존 RAG가 검색 문서를 무분별하게 사용하는 것과 달리, SELF-RAG는 생성 결과의 근거성과 완결성을 함께 고려함.
---

### 3.1 문제 정의 및 전체 구조

![image.png](attachment:8cb06b0c-ce31-4029-982a-5ddeefee6f1e:image.png)

입력 xxx가 주어졌을 때, 모델 MMM은 여러 개의 출력 세그먼트로 구성된 텍스트 y=[y1,…,yT]y = [y_1, \dots, y_T]y=[y1,…,yT]를 순차적으로 생성함. 각 세그먼트는 일반 텍스트 토큰뿐 아니라 reflection token을 포함할 수 있음.

추론 시 SELF-RAG의 동작 방식은 다음과 같음.

모델은 현재 입력과 이전 생성 결과를 바탕으로 먼저 **검색이 필요한지 여부를 판단**함. 검색이 필요하지 않다면 일반 언어 모델처럼 다음 세그먼트를 생성함. 검색이 필요하다고 판단되면, 검색을 수행한 뒤 각 문서에 대해 관련성(relevance)과 지원 여부(support)를 평가하면서 답변을 생성함. 이후 전체 응답에 대한 유용성까지 평가하여 최종 출력을 선택함.

이 과정에서 SELF-RAG는 여러 검색 문서를 **병렬적으로 처리**하고, 스스로 생성한 reflection token을 활용해 불필요한 문서를 걸러내거나 생성 결과를 제어함.

---

### 3.2 SELF-RAG 학습 방식

SELF-RAG는 두 개의 모델을 학습함. 하나는 **critic 모델(C)**, 다른 하나는 **generator 모델(M)**임.

### 3.2.1 Critic 모델 학습

Reflection token을 사람이 직접 라벨링하는 것은 비용이 크기 때문에, GPT-4를 활용해 supervision 데이터를 생성함. GPT-4에 특정 지시문과 예시를 제공하여, 검색 필요 여부나 생성 결과의 품질을 판단하는 reflection token을 생성하게 하고, 이를 기반으로 critic 모델을 학습함.

Critic 모델은 입력과 출력 쌍을 받아 적절한 reflection token을 예측하도록 학습되며, GPT-4의 판단과 높은 수준의 일치도를 보임. 이를 통해, 추론 단계에서는 외부 대형 모델에 의존하지 않고도 reflection token을 생성할 수 있도록 함.

### 3.2.2 Generator 모델 학습

Generator 모델은 실제 SELF-RAG의 추론 과정을 모사한 데이터로 학습됨.

입력–출력 쌍에 대해, critic과 retriever를 사용해 검색 여부 판단, 문서 관련성 평가, 지원 여부 평가, 전체 유용성 평가를 수행하고, 이 과정에서 생성된 reflection token을 원래 출력에 삽입함.

이렇게 구성된 데이터셋을 사용해, generator 모델은 **일반적인 다음 토큰 예측 방식**으로 학습되며, 텍스트와 reflection token을 모두 예측하도록 학습됨. 이 과정은 RLHF와 달리 추가적인 강화학습 없이 수행되어, 학습 비용을 크게 줄임.

---

### 3.3 SELF-RAG 추론 단계 제어

SELF-RAG는 추론 단계에서 reflection token을 활용해 **모델 동작을 제어**할 수 있음.

사실성이 중요한 태스크에서는 검색을 더 자주 수행하도록 유도할 수 있고, 창의성이 중요한 개방형 생성 태스크에서는 검색 빈도를 낮출 수 있음.

검색 여부는 Retrieve 토큰의 예측 확률이 특정 임계값을 넘는지에 따라 결정할 수 있음. 검색이 수행되면, 여러 검색 문서를 대상으로 병렬적으로 후보 응답을 생성하고, critique token을 기반으로 세그먼트 단위의 beam search를 수행함.

각 후보 응답은 문서 관련성, 지원 여부, 전체 유용성 등 여러 기준을 반영한 점수로 평가되며, 이 점수의 가중치는 추론 시 조절 가능함. 이를 통해, 추가 학습 없이도 사실성·완결성·유용성 간의 trade-off를 조정할 수 있음.

기존 RLHF 기반 접근법이 학습 단계에서 모델 행동을 변경하는 데 초점을 둔다면, SELF-RAG는 **추론 단계에서의 제어 가능성**을 제공한다는 점에서 차별화됨.

---

## 4. Self-RAG 학습 (Training Self-RAG)

### 4.1 비평가 모델 학습 (Training the Critic Model)

**학습 목표**

- 입력 질문 `x`, 검색 문서 `c`, 생성 답변 `y`를 평가
- 비평 토큰(Reflection Tokens) `r`을 **적절한 위치에 삽입**하는 능력 학습

**데이터 수집 과정**

- **GPT-4 활용**
    - 비평 토큰 유형별 소수의 고품질 주석 데이터(Few-shot)
    - 명시적인 지시문(Instruction)과 함께 입력
- **프롬프트 구성 예시**
    - “제공된 문서가 질문에 답하는 데 유용한가?” → `IsUSE`
    - “답변의 모든 문장이 문서에 의해 뒷받침되는가?” → `IsSUP`
        
        → GPT-4가 비평 토큰을 직접 생성하도록 유도
        
- **데이터 증류 (Distillation)**
    - GPT-4 생성 결과를 표준 언어 모델링 데이터셋으로 변환
    - 소형 모델 (예: Llama-2-7B)을 비평가 모델 `C`로 미세조정

**학습 수식**

$L_{critic} = E_{(x, c, y, r) \sim D_{critic}} [-\log P(r | x, c, y)]$

- **수식 정의**
    
    주어진 **질문** x, **문맥** c, **답변** y 조건 하에서  
    올바른 **비평 토큰** r 이 생성될 확률을 최대화하기 위한 **손실 함수**임.
    
- **구성 요소의 의미**
    - **`x, c, y, r`**
        
        각각 **질문 (Input)**, **문맥 (Context)**, **답변 (Output)**, **비평 토큰 (Reflection Token)** 을 의미함.
        
    - **`E`**
        
        비평가 모델용 학습 데이터셋 D 전체에 대한 **기대값(평균)** 을 계산함.
        
    - **`−log P`**
        
        **음의 로그 우도 (Negative Log-Likelihood)**
        
        모델이 예측한 토큰이 실제 정답 r 과 다를수록 손실값이 커지는 구조임.
        
- **학습의 목적**
    - **GPT-4가 생성한 고품질 비평 데이터를 소형 모델(LLaMA 등)이 그대로 복제(Distillation)** 하도록 만드는 것이 핵심임.
    - 이 과정을 통해 소형 모델인 **비평가 모델 C** 가 **인간 또는 GPT-4 수준의 판단 능력**을 갖추도록 유도함.

**특이 사항**

- 비평가 모델 `C`는 다음을 함께 학습함
    - 문서 기반 평가
    - **검색이 필요한 시점에서 [Retrieve] 토큰을 언제 생성할지 판단**
## 4.2 생성 모델 학습 (Training the Generator Model)

### 학습 목표

- 외부 비평가 없이도  
  → 스스로 비평 토큰과 답변을 동시에 생성하는 모델 `M` 구축

### 데이터 보강 (Data Augmentation)

- 원본 말뭉치 `D`의 각 샘플에 대해 비평가 모델 `C` 실행
- 처리 흐름
  1. 질문 `x` 뒤에 [Retrieve] 토큰 삽입 여부 판단
  2. 검색된 문서 `d` 획득
  3. 텍스트 사이에 다음 토큰 삽입
     - [IsREL] : 문서-질문 관련성
     - [IsSUP] : 문서 기반 근거 수준 (문서를 충실하게 참고하고 있는지 / 할루시네이션)
     - [IsUSE] : 답변 전반적 유용성

### 전체 생성 시퀀스 예시

[질문] 비타민 C가 감기 예방에 도움이 될까?

[Retrieve:Yes]  
(← 모델이 검색이 필요하다고 판단함)

[문서 1]  
최근 메타 분석 결과에 따르면, 일반적인 상황에서 비타민 C 복용이 감기 발병률을 유의미하게 낮추지는 못하지만, 운동선수와 같은 특정 집단에서는 예방 효과가 있는 것으로 나타남.  
[IsREL:Relevant]  
(← 문서 1이 질문과 직접적인 관련이 있음을 표시함)

[생성 답변]  
일반인에게 비타민 C가 감기를 완전히 예방해주는 효과는 크지 않지만, 격렬한 신체 활동을 하는 사람들에게는 도움이 될 수 있습니다.  
[IsSUP:Fully supported]  
(← 방금 생성한 문장이 문서 1에 완전히 근거하고 있음을 표시함)

[IsUSE:5]  
(← 이 답변이 사용자 질문에 매우 유용함을 점수로 표시함)

### 학습 방식

- 보강된 데이터셋 `D_gen` 사용
- 표준 Next-token prediction 방식으로 학습
- 중요 포인트
  - 비평 토큰을 특수 토큰이 아닌 일반 어휘처럼 처리
  - 문맥 속에서 자연스럽게 비평 토큰이 등장하도록 학습
  - 검색이 필요 없는 일반 텍스트 데이터도 함께 학습  
    → 불필요한 검색 남용 방지

---

## 4.3 Self-RAG 추론 (Self-RAG Inference)

### 1) 적응형 검색 (Adaptive Retrieval)

- 모델이 [Retrieve] 토큰을 생성할 확률을 지속적으로 모니터링
- 임계치 기반 검색 호출

P([Retrieve] = Yes | x, y_prev) > τ

- 조건을 만족할 때만 Retriever 호출
- 효과
  - 불필요한 검색 비용 감소
  - 필요한 순간에만 외부 지식 보충

### 2) 트리 디코딩 (Tree Decoding)

- 검색된 K개의 문서 각각에 대해
  - 병렬적으로 답변 후보 생성
- 각 경로에서 비평 토큰이 실시간 생성
- 결과적으로 Beam Search와 유사한 구조 형성

### 3) 비평 토큰 기반 랭킹 수식 (Ranking Formula)

Score(y, d) =  
P(IsREL)  
+ w1 · P(IsSUP)  
+ w2 · P(IsUSE)

- P(G): 비평 토큰 G가 최고 등급을 가질 확률  
  (예: Fully supported)
- w1, w2: 태스크 목적에 따라 조절하는 가중치

---

## 5. 실험 분석 (Experiments & Analysis)

### 5.1 주요 벤치마크 및 지표

#### Closed-set QA (폐쇄형 질의응답)

- ARC-Challenge
- OpenbookQA
- 평가 지표: Accuracy

#### Short-form QA (단답형 질의응답)

- PopQA
- TriviaQA
- 평가 지표: Accuracy (Exact Match 등)

#### Long-form QA (장문형 질의응답)

- ASQA (Ambiguous QA)
- 평가 지표
  - 사실성 (Factuality)
  - 인용 정확도 (Citation Accuracy)

#### Fact Checking (사실 확인)

- FactScore
- 평가 지표: Fine-grained Factuality

### 5.2 실험 결과 요약

- 전체 성능
  - Self-RAG 7B 모델이 Llama-2-13B 및 일반 RAG 대비 대부분의 지표에서 우수
- FactScore
  - 기존 RAG 대비 약 10%p 이상 사실성 향상
- 검색 효율
  - Retrieve 기반 적응형 검색으로 검색 횟수 최적화

#### PubHealth 결과 해석

1. 노이즈에 대한 높은 민감도  
   - 검색 임계치가 낮을수록 오히려 성능 저하 발생
   - 의료/보건 도메인에서는 소수의 고품질 문서가 중요

2. 학습 데이터 포화 및 역효과 (Negative Transfer)  
   - 데이터 규모 증가 시 성능 정체 또는 소폭 하락
   - 전문 도메인에서는 정제된 고품질 데이터가 더 중요함을 시사

---

## Appendix

### Appendix A. 비평 토큰 정의 (Token Schema)

[Retrieve] : 검색 필요 여부  
- Yes / No / Continue

[IsREL] : 문서-질문 관련성  
- Relevant / Irrelevant

[IsSUP] : 답변의 문서 기반성  
- Fully supported / Partially supported / No support

[IsUSE] : 답변 전반적 품질  
- -1 ~ 5

### Appendix B. 학습 하이퍼파라미터

Base Model: Llama-2-7B, 13B  
Learning Rate: 2e-5  
Batch Size: 128  
Context Length: 4096  
Epochs: 3

### Appendix C. 비평가 모델 프롬프트 예시

질문 x와 문서 d가 주어졌을 때,  
문서가 질문에 직접적인 정보를 제공하면 Relevant,  
그렇지 않으면 Irrelevant를 출력하라.

### Appendix D. 점수 가중치 설정 예시

- 사실성 중심 태스크  
  - w_IsSUP = 1.0
- 유창성·유용성 중심 태스크  
  - w_IsUSE = 0.8