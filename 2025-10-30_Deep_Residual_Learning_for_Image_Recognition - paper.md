# 1. 논문 요약

## (1) 문제의식

2015년 이전의 딥러닝은 깊을수록 성능이 높다는 의견 아래 발전해 왔으나, VGGNet이나 GoogLeNet처럼 층수를 늘린 모델들은 어느 순간부터 오히려 깊이가 깊을수록 훈련 오차가 증가하는 현상을 보였다. 이것이 바로 **Degradation Problem**이다.

이에 He et al.(2015)는 단순한 gradient vanishing이 아니라, 네트워크가 **항등 함수(Identity Mapping)** 를 표현하기 어려워서 발생하는 최적화 실패라고 분석했다.

즉, 모델이 복잡한 함수를 학습하는 도중 '아무 변화도 일어나지 않아야 할 경우(=항등)'를 정확히 학습하지 못한다는 것이다.

이 문제를 해결하기 위해 저자들은 **Residual Learning**이라는 새로운 학습 프레임워크를 제시했다.

---

## (2) 핵심 아이디어 - 잔차 학습(Residual Learning)

ResNet은 기존 CNN이 `H(x)`라는 전체 변환 함수를 직접 학습하는 대신, 입력과의 차이인 잔차 `F(x) = H(x) - x`를 학습하도록 구조를 바꿨다.

즉, 모델은 전체 함수를 새로 만드는 대신, 입력 대비 얼마나 변화를 주어야 하는가를 학습한다.

이 단순한 재정의는 **Gradient 흐름의 병목을 제거**하고, 학습 안정성을 획기적으로 높였다.  
만약 최적의 함수가 단순한 항등 함수라면, 모델은 `F(x) → 0`으로만 수렴하면 되므로 학습이 훨씬 단순해진다.

---

## (3) 이론적 배경 - 수치해석의 잔차 개념

저자는 '잔차'라는 아이디어를 공학적 배경에서 가져왔다.

저수준 비전이나 편미분 방정식(PDE) 해법에서는 이미 **Multigrid Method**나 **Hierarchical Basis Preconditioning**과 같은 방식으로 잔차를 반복적으로 계산하며 빠르게 수렴시키는 방법이 존재했다.

ResNet은 이러한 **잔차 최소화(residual minimization)** 접근을 신경망 최적화 과정에 그대로 도입한 것이다.  
즉, 복잡한 함수를 직접 근사하는 대신, 입력 대비 오차를 줄이는 방향으로 학습을 단순화시킨 셈이다.

---

## (4) 구조 설계 - Residual Block과 Bottleneck Block

ResNet의 기본 단위는 **Residual Block**이다.  
이는 두 개의 3×3 convolution layer와 **항등 연결(identity shortcut)** 로 구성된다.

입력과 출력의 차원이 다른 경우 1×1 convolution(=linear projection)을 통해 크기를 맞춘다.  
이 기본 형태는 **ResNet-18**과 **ResNet-34**에 사용된다.

하지만 깊이를 100층 이상으로 확장하면 연산량이 폭발하므로, 저자들은 **Residual Block이 아닌 Bottleneck Block**을 설계했다.  
이는 `1×1 → 3×3 → 1×1` 구조로 중간 채널 수를 줄여 효율성을 확보한 것이다.

| 모델 | 블록 구성 | 특징 |
|------|------------|------|
| ResNet-18 / 34 | 3×3 → 3×3 | Basic Block |
| ResNet-50 / 101 / 152 | 1×1 → 3×3 → 1×1 | Bottleneck Block (효율적 구조) |

---

## (5) 실험 결과 - ImageNet과 CIFAR-10 검증

ImageNet에서 ResNet은 깊이에 따라 성능이 꾸준히 향상되었다.

| 모델 | 층 수 | Top-1 Error | 비고 |
|------|--------|--------------|------|
| Plain CNN | 34 | 27.9% | Degradation 발생 |
| ResNet-34 | 34 | 25.0% | 안정적 학습 |
| ResNet-50 | 50 | 24.0% | Bottleneck 도입 |
| ResNet-152 | 152 | **23.0%** | ILSVRC 2015 1위 |

또한 CIFAR-10에서는 1000층 이상으로 확장해도 학습이 실패하지 않았다.  
이는 Residual 구조가 깊이에 독립적인 최적화 안정성을 제공한다는 것을 실험적으로 입증한 사례다.

---

## (6) 확장 실험 - Object Detection과 Segmentation

ResNet은 단순 분류를 넘어 **Faster R-CNN의 Backbone**으로 적용되며, VGG-16 대비 mAP가 3% 이상 향상되었다.

이 결과는 **Residual Feature**가 고차원적 시각 과제에서도 효과적임을 보여준다.

---

## (7) 결론

ResNet은 단순한 모델 아키텍처가 아니라 신경망 학습의 **공통 언어**를 제시했다.

모든 현대 모델은 형태만 다를 뿐, 본질적으로 다음 식 위에서 작동한다.

`y = F(x) + x`

즉, ResNet은 딥러닝이 **'변화량을 학습하는 체계'** 로 진화하는 출발점이다.

---

# 2. 모델의 목적에 따라 달라지는 잔차 함수

ResNet의 `F(x)`는 단순히 Conv 연산이었지만, 이후 모델들은 자신의 목적에 맞게 잔차의 정의를 새롭게 확장했다.

결과적으로 딥러닝은 **잔차의 형태에 따라 분화된 생태계**로 발전했다.

| 계열 | 잔차 함수의 정의 | 잔차의 의미 | 대표 모델 |
|------|------------------|--------------|------------|
| 구조적 Residual | Conv + BN + ReLU | 공간적 특징 보정 | ResNet, ResNeXt, DenseNet |
| 주의 기반 Residual | Conv + Attention | 채널, 공간 중요도 조정 | SENet, CBAM |
| 관계 기반 Residual | Self-Attention + MLP | 토큰 간 상호 관계 변화 | Transformer, ViT |
| 확률 기반 Residual | Noise Estimation | 데이터 분포 - 노이즈 차이 | Diffusion, DDPM |
| 기하 기반 Residual | Projection Error | 좌표, 시점 간 오차 | NeRF, 3DGS |

이 모든 모델은 공통적으로 **'입력 대비 변화량을 학습한다'**는 철학을 공유하지만,  
그 변화량의 대상이 공간 / 관계 / 확률 / 기하 / 주의(attention)로 달라진 것이다.

---

# 3. EfficientNet: ResNet을 바탕으로 한 최신 딥러닝 기술

2019년 Google Brain에서는 ResNet의 철학을 계승하여 **효율적 확장(Compound Scaling)** 문제를 다뤘다.

ResNet이 깊이(depth)에 집중했다면, EfficientNet은 **깊이, 너비, 해상도**를 동시에 확장하는 방식을 제시했다.

(α, β, γ: scaling coefficient, φ: 자원 제약 지수)

이 방식은 모델의 파라미터 수를 단순히 늘리는 대신, 세 축의 균형을 맞춰 효율성을 극대화했다.

| 모델 | Params(M) | FLOPs(B) | Top-1 Acc. |
|-------|------------|----------|-------------|
| ResNet-152 | 60.2 | 11.5 | 77.0% |
| EfficientNet-B7 | 66.0 | 37.0 | **84.3%** |

효율은 최대 **8배 이상 향상**되었다.  
ResNet이 깊이에 치중한 형태라면, EfficientNet은 **균형 있게 늘리는 법**을 보여준 셈이다.

EfficientNet 역시 **Residual Block**을 기본 단위로 사용하지만, 내부 `F(x)`의 정의를 발전시켰다.  
이는 **Depthwise Separable Convolution**과 **Squeeze-and-Excitation(SE)** 모듈을 결합하여 정보 손실 없이 효율을 높였다.

즉, `y = F(x) + x`라는 ResNet의 수식은 그대로 유지하면서,  
그 안의 `F(x)`를 더 효율적이고 세밀한 연산으로 교체한 것이다.

---

# 4. ResNet → EfficientNet, 잔차 학습의 발전

| 구분 | ResNet(2015) | EfficientNet(2019) |
|------|---------------|--------------------|
| 학습 대상 | Residual Mapping `F(x)` | Compound Scaling `(α, β, γ)` |
| 확장 축 | Depth 중심 | Depth + Width + Resolution |
| 목표 | Gradient 안정화 | 효율 극대화 |
| 철학 | 변화량 학습 | 자원 최적화 |

ResNet은 단순한 구조 혁신이 아니라, **학습을 잔차의 문제로 재정의한 전환점**이다.  
이후 ResNet을 베이스로 한 모든 모델(Transformer, Diffusion, NeRF, EfficientNet)은 모두  
**입력과의 차이를 학습하는 구조**라는 동일한 원리를 공유한다.

---

**출처:**  
(Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun), *Deep Residual Learning for Image Recognition* (2015.10)  
<https://arxiv.org/abs/1512.03385>
