# 1. 서론(GAN은 왜 중요한가)

Ian Goodfellow의 2016 GAN Tutorial은 생성 모델링 분야 전체를 관통하는 중요한 논문이다. 이 논문은 GAN의 구조만 설명하는 것이 아니라, 기존 생성 모델들의 한계와 GAN이 등장해야 했던 필연적 이유를 체계적으로 제시한다.

기존 explicit likelihood 기반 생성 모델들은 이론적으로 정교했지만 실제 고차원 데이터, 특히 이미지를 다루기에는 구조적, 계산적 제약이 컸다.

GAN은 이러한 제약을 벗어난 implicit generative model로, 확률 밀도를 직접 정의하거나 계산하지 않고 샘플을 생성하는 과정만 학습한다.

이 접근 방식은 이미지 생성 품질, 다양성, 표현력 측면에서 기존 방식보다 훨씬 유리한 방향으로 발전하게 한다.

---

# 2. 생성 모델링의 전체적 구도 (Expllicit vs. Implicit)

모든 생성 모델의 목표는 **[데이터 분포 pdata(x)를 잘 모사하는 모델 분포 pmodel(x)를 만드는 것]**이다. 이를 달성하는 방식은 크게 두 가지로 볼 수 있다.

## ① Explicit Models
- 확률 밀도 함수 p(x)를 명시적으로 정의  
- likelihood를 직접 계산하거나 근사  
- **장점:** 이론적으로 명확하고 확률적 일관성 보장  
- **단점:** 고차원에서는 구조 제약 및 계산 복잡성 증가  

## ② Implicit Models
- p(x)를 정의하지 않음  
- 오직 샘플로 G(z)만 정의  
- **장점:** 구조적 제약이 적어 복잡한 데이터 잘 모델링  
- **단점:** likelihood 계산 불가로 평가 어려움  

GAN은 이 중 **Implicit Model**이다.  
이는 GAN이 가지는 강력한 표현력의 이유일뿐만 아니라, 동시에 학습이 까다로운 이유이기도 하다.

---

# 3. Explicit Models

Explicit 모델은 분포를 직접 계산하거나 근사하려 하기에 이론적으로 간결하다. 그러나 고차원의 이미지 생성에 적용하면 많은 제약이 드러난다. Explicit 모델은 크게 두 가지로 나뉜다.

---

## 3.1 Tractable Explicit Models — 정확한 likelihood 계산 가능

### (1) FVBN(Fully Visible Belief Networks)

FVBN은 chain rule에 따라 데이터를 다음과 같이 분해한다.

#### ① 장점
- likelihood 정확히 계산 가능  
- 구조적 직관성이 높음  

#### ② 단점
- 샘플 생성이 무조건 순차적이므로 병렬화 불가능하며, 매우 느림  
- 고차원 이미지 생성에 부적합함  

WaveNet도 이 구조를 기반으로 했으며 음성 품질은 매우 좋았으나 생성 속도는 너무 느렸다.

---

### (2) Normalizing Flows

Flow 모델은 invertible mapping을 이용해 모든 점을 latent와 1:1로 대응시킨다.  
Jacobian determinant를 이용해 exact likelihood를 계산할 수 있다.

#### ① 장점
- 정확한 likelihood  
- invertible이므로 양방향 전환 가능  

#### ② 단점
- 반드시 invertible이어야 한다.  
- latent와 데이터 차원이 동일해야 한다.  

결과적으로 **구조적 제약으로 인하여 표현력이 제한**된다.  
즉, 이론적으로는 우수하지만 실용적 유연성이 부족하다고 보면 된다.

---

## 3.2 Intractable Explicit Models

### (1) VAE (Variational Autoencoders)

VAE는 exact likelihood를 계산할 수 없어 다음의 ELBO를 최적화한다.  
문제는 ELBO가 느슨하면 실제 likelihood와 간극이 발생하고, latent 주변을 평균화하면서 **블러 현상**이 나타난다는 것이다.

즉, 안정적이고 이론적으로 깔끔하지만 생산 품질이 낮아지는 경향이 있다.

---

### (2) Boltzmann Machines

Boltzmann Machines 모델은 에너지 기반 모델로 학습에 MCMC가 필수이다.  
그러나 고차원에서 MCMC는 수렴의 판별이 불가능하기 때문에 계산량이 지나치게 크고 실용성이 없다.

따라서 현대 딥러닝에서는 사실상 사용되지 않는다.

---

# 4. Implicit Models (분포를 직접 정의하지 않는 생성 모델)

Implicit 모델은 확률 분포를 직접 계산하는 대신, 다음과 같은 샘플러를 정의한다.

이 방식은 분포 계산을 포기하는 대신 **구조적 유연성**을 얻고, **복잡한 데이터 생성에 매우 강력한 성능**을 발휘한다.  
GAN은 이 카테고리의 대표 모델이다.

---

# 5. GAN의 구조와 이론

GAN은 두 네트워크가 경쟁하는 minimax 게임이다.

- **Generator G:** 진짜처럼 보이는 가짜 샘플 생성  
- **Discriminator D:** 진짜 vs. 가짜 구분  

---

## 5.1 기본 Objective
G는 D를 속이고, D는 G를 이기려 한다.

## 5.2 Optimal Discrimanator
사진 설명을 입력하세요.  
GAN은 사실상 **density ratio를 학습**한다고 할 수 있다.

## 5.3 JS Divergence 문제

GAN의 objective는 JS divergence를 최소화한다.  
그러나 두 분포의 support가 겹치지 않으면 JS divergence의 gradient는 0이 된다.

1) 학습 초기 G가 엉성하면  
2) D는 완벽한 판별이 가능하나  
3) G는 gradient를 받지 못하므로  

결과적으로 GAN은 초기 학습에서 실패하게 된다.  
이를 **JS Divergence 문제**라고 한다.

---

# 6. Non-Saturating Loss

기본 minimax에서 G는 다음을 최소화한다.  
사진 설명을 입력하세요.

초기에는 D(G(x))가 0으로 수렴하면 gradient가 거의 0이 된다.

이를 해결하기 위해 사용되는 것이 **non-saturating loss**다.  
사진 설명을 입력하세요.

만약 D(G(z)) = 0이면 log0 = -∞가 되어 gradient가 매우 커져  
G는 강한 학습 신호를 받는다.

즉, 초기 학습 실패 문제를 완화할 수 있다.

---

# 7. GAN 안정화 테크닉

| 기법 | 목적 | 설명 | 효과 |
|------|------|------|-------|
| Feature Matching | 안정화 | D의 feature 통계를 모방 | 진동 완화, mode collapse 감소 |
| Minibatch Discimination | 다양성 | 배치 간 차이를 D가 학습 | 동일 샘플 반복 생성 차단 |
| Historical Averaging | 진동 억제 | 파라미터가 과거 평균에서 크게 벗어나지 않도록 규제 | 학습 안정 |
| One-sided Label Smoothing | D 안정 | real을 1대신 0.9로 | G의 gradient 개선 |
| Virtual BatchNorm | 입력 변동 완화 | 고정 reference batch 기준 BN | 생성 안정성 증가 |
| Instance Noise | gradient 확보 | real/fake에 노이즈 추가 | 초기 D 과강화 방지 |

---

# 8. GAN 평가 방법

| 지표 | 목적 | 장점 | 단점 | 현재 위치 |
|------|------|------|------|------------|
| IS | 품질, 다양성 측정 | 계산 간단 | real 분포를 고려 안 함 | 거의 사용 안 함 |
| FID | real/fake feature 분포 비교 | 품질+다양성 모두 반영 | Inception feature 의존 | 현대 표준 |
| Birthday Paradox Test | mode collpase 탐지 | support size 추정 가능 | 간접적 | 보조적 사용 |

---

# 9. 왜 GAN은 불안정한가 (Minimax 구조의 본질)

GAN은 minimax 게임이다.  
전통적 gradient descent 방식은 minimax 문제에 적합하지 않다.

x*y와 같은 단순한 minimax 게임조차 simultaneous gradient descent를 적용하면  
수렴하지 않고 원점 근처를 **회전**한다.

그러므로 GAN이 흔히 보이는  
- 진동  
- 발산  
- mode collapse  

의 근본 원인은 **minimax 구조 자체**라고 할 수 있다.

---

# 10. GAN의 확장

| 영역 | 기법 | 핵심 아이디어 | 효과 |
|------|------|----------------|-------|
| Semi-supervised Learning | Semi-Supervised GAN | D를 K+1 클래스 분류기로 확장 | 적은 라벨로도 높은 정확도 |
| Representation Learning | GAN Feature Extraction | D feature 활용 | 강력한 unsupervised representation |
| Image Editing | GAN Inversion | 이미지로 latent를 역추적 | 속성 조작 가능 |
| Disentanglement | InfoGAN | mutual information 극대화 | 의미 있는 factor 학습 |
| High-resolution Generation | Progressive GAN | 점진적 해상도 확장 | 고해상도 생성 |
| Stability | Unrolled GAN | future D update 고려 | mode collapse 감소 |

---

# 11. 결론

GAN Tutorial은 explicit 모델과 implicit 모델의 대립구조, likelihood 기반 모델의 한계, GAN의 density ratio 관점, minimax 구조적 불안정성, 평가 지표, representation learning 가능성 등 생성 모델링의 핵심 개념을 모두 보여준다.

하지만 GAN은 여전히 mode collapse, gradient instabilty, hyperparameter 민감성 등 구조적 한계를 안고 있다.

이 문제를 정면으로 해결하기 위해 등장한 모델 계열이 **Diffusion Models**이다.

likelihood 기반, 안정적 최적화, mode collpase 비존재, 단순한 목표 구조 등의 장점으로 Diffusion Models은 현재 생성 모델 분야의 주류가 되었다.

그렇기에 다음에 논문 스터디의 논문을 선정할 차례가 왔을 때 다 같이 Diffusion 모델에 관해 읽어보고 싶다.

---
출처: https://arxiv.org/abs/1701.00160
NIPS 2016 Tutorial: Generative Adversarial Networks, Ian Goodfellow, [Submitted on 31 Dec 2016 (v1), last revised 3 Apr 2017 (this version, v4)]
