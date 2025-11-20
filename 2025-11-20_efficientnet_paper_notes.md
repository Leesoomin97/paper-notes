# EfficientNet 논문 정리  
**EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**

---

## 1. Abstract

EfficientNet은 CNN을 확장할 때 전통적으로 사용하던 방식(깊이, 너비, 해상도 중 선택적 증가)이 비효율적이라는 점을 실험적으로 보여주고, 세 축을 단일 비율로 동시에 확장하는 방법(compound scaling)을 제안한다.

NAS로 설계한 EfficientNet-B0을 기반으로 이 스케일링을 적용해 B1~B7까지 모델을 확장했고, ImageNet에서 기존 SOTA CNN보다 8.4배 더 작고 6.1배 빠른 모델로 동일 성능을 달성했다.

---

## 2. Related Work

EfficientNet은 다음 세 가지 스케일링 방식이 모두 불완전하다는 점을 짚는다.

### (1) Depth Scaling
Resnet 계열처럼 layer를 깊게 쌓는 방식을 말한다. 이는 효과는 있지만 일정 깊이 이상에서는 gradient 흐름 약화로 성능 향상이 정체된다.

### (2) Width Scaling
Wide-ResNet, MobileNet처럼 채널 수를 늘리는 방식을 말한다. 그러나 shallow-wide 구조에서 고수준 feature 형상이 어렵고, 파라미터만 급증한다.

### (3) Resolution Scaling
입력 해상도를 키워 더 많은 픽셀 정보를 확보한다. 문제는 FLOPs 증가가 r^2에 비례해 연산 비용이 급증하는데 반해, 정확도 상승은 매우 제한적이다.

또한 NASNet, AmoebaNet 같은 NAS 기반 구조는 높은 성능을 보이지만 모델 규모와 연산량이 크고 학습 비용이 매우 높다는 점도 지적한다.

---

## 3. Compound Model Scaling

EfficientNet의 핵심적인 이론적 기여는 CNN 스케일링을 수식 기반으로 체계화한 점이다. 이 장은 다음 세 부분으로 구성된다.

---

### (1) Problem Formulation

CNN의 연산량은 다음과 같이 근사된다.

- d: depth – block 반복 횟수  
- w: width – 채널 수  
- r: resolution – 입력 이미지 크기

따라서 depth는 선형 증가이지만, width와 resolution은 비용을 제곱으로 증가시킨다.

문제는 다음과 같다.

- 기존 모델들은 세 축 중 하나만 증가하며  
- 이 방식은 FLOPs 대비 accuracy 향상 효율이 낮고  
- 따라서 세 축을 모두 고려한 스케일링 전략이 필요하다.

이를 해결하기 위해 논문은 세 축을 동시에 확장하는 수식 기반 전략(compound scaling)을 제안한다.

---

### (2) Scaling Dimensions

논문 Figure 3은 각 축을 단독으로 증가시켰을 때의 비효율성을 실험으로 보여준다.

#### ① Depth 증가 (d-only)
- 초반엔 accuracy 상승이 있으나 일정 깊이 이상에서는 gain 거의 없음  
- gradient 흐름 약화 및 학습 난이도 증가

#### ② Width 증가 (w-only)
- 파라미터는 급증하지만 accuracy 증가는 미미  
- shallow-wide 구조 특성상 high-level feature 표현에 한계 발생

#### ③ Resolution 증가 (r-only)
- resolution 두 배 증가 시 FLOPs는 r^2 = 4배 증가  
- accuracy 상승은 매우 제한적

결과적으로 단일 축 증가 방식은 모든 경우 비용 대비 효율이 낮음을 알 수 있다.

---

### (3) Compound Scaling

단일 scaling의 비효율성을 극복하기 위해 논문은 다음 공식을 제안한다.

- φ를 1 증가시키면 FLOPs는 약 2배 증가  
- 즉, 모델 규모를 정해진 자원 예산 내에서 일정하게 컨트롤 가능  
- 세 축을 동시에, 서로 다른 증가율로 증가시키며 최적 비율 유지

논문은 grid search를 통해 다음 계수를 선택했다.

- α = 1.2 (depth 증가율)  
- β = 1.1 (width 증가율)  
- γ = 1.15 (resolution 증가율)

이 값들은 제곱 항 때문에 width, resolution 증가율이 깊이보다 더 작게 설정되었다는 점이 특징이다.

---

## 4. EfficientNet Architecture

EfficientNet-B0은 NAS 기반으로 설계되었으며 다음 블록 구조를 가진다.

- MBConv(inverted bottleneck 구조)  
- expansion ratio 6 적용  
- depthwise convolution  
- squeeze-and-excitation(SE)  
- Swish(SiLU) activation  

B1~B7은 B0의 구조를 그대로 유지하고, compound scaling 공식만 적용해 depth/width/resolution을 동시에 확장한 모델군이다.  
(B4: input 380×380, B7: input 600×600)

구조를 건드리지 않으면서도 모델 규모만 수식 기반으로 확장한 것이 EfficientNet의 핵심이다.

---

## 5. Experiments

### (1) ImageNet Classification
- EfficientNet-B7: **84.3% Top-1**  
- 동일 성능의 GPipe는 557M 파라미터  
- B7은 66M 파라미터, 즉 **8.4배 더 작고 학습 속도 6.1배 빠름**

### (2) FLOPs Efficiency

| 모델        | Params | FLOPs | Top-1 |
|-------------|--------|--------|--------|
| B4          | 19M    | 4.2B   | 83.0% |
| ResNet-152  | 60M    | 11.5B  | 78.3% |

- FLOPs 3배 감소  
- 파라미터 3배 감소  
- 정확도는 4.7% 상승

### (3) Transfer Learning
Flowers, Cars, CIFAR-100 등의 전이 학습에서도 기존 모델 대비 평균보다 적은 파라미터로 SOTA 달성.

---

## 6. (추가공부) Compound Scaling에서 폭발적 모델 증가 문제: 제곱 증가항의 영향

논문을 보던 중 depth와 달리 width와 resolution는 제곱으로 증가하는 것을 보며 이러면 모델의 scale이 너무 커질 뿐만 아니라 균형적인 증가가 어려운 게 아닐까 의문이 들어 추가적으로 조사해보았다.

Compound Scaling 공식에서 width와 resolution은 FLOPs에 제곱항으로 기여한다.  
이 구조는 세 축을 균형적으로 늘릴수록 FLOPs가 빠르게 증가하는 문제를 갖는다.

예를 들어, d, w, r을 모두 1.2배로 키우면:

1.2 * 1.2^2 * 1.2^2 = 약 2.49배

즉, 20%씩 늘려도 FLOPs는 2.49배가 되며, 균형적 증가가 자칫 모델 폭증을 초래할 수 있다.

EfficientNet이 이를 해결한 방식은 다음과 같다.

### ① α, β, γ을 매우 작은 계수(1.x)로 설정
- width와 resolution의 증가율을 depth보다 더 낮게 설정  
- 제곱항의 폭발적 증가를 억제하기 위한 조치

### ② FLOPs = 2^φ 제약으로 증가율을 통제
- compound scaling 자체가 FLOPs 변화를 정교하게 관리하는 방식

### ③ 균형적 증가 ≠ 동일율 증가
- EfficientNet의 목표는 세 축을 같은 속도로 키우는 것이 아니라  
  각 축이 만드는 비용(선형/제곱)을 고려해  
  현실적 하드웨어 한계 내에서 가장 합리적인 증가율 조합을 찾는 것

따라서 compound scalling은 이론적 이상형이 아니라  
**실제 GPU 자원 제약 속에서 최적의 균형을 찾은 실용 확장 방식**이라고 이해할 수 있다.

---

## 6. Discussion & Conclusion

EfficientNet은 CNN scaling을 경험적 조정에서 벗어나 수식, 실험 기반의 체계적 접근으로 끌어올린 모델이다.

- depth, width, resolution의 상호 의존 관계를 밝히고  
- FLOPs 증가 구조를 기반으로 확장률을 설계했으며  
- NAS baseline + compound scaling으로 작은 모델부터 큰 모델까지 효율적으로 확장했다.

이 논문은 이후 ConvNext, Swin, ViT 기반 모델들이  
모델군을 여러 scale로 제공하는데 영향을 준,  
스케일링 전략의 중요한 전환점이라고 볼 수 있다.

---

