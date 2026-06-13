# Lorenz DA Lab 학습 로드맵

이 문서는 Lorenz DA Lab 프로젝트의 전체 학습 순서를 정리한 문서입니다.

목표는 Lorenz-63 모델을 이용해 수치모델, 자료동화, tangent linear model, adjoint model, 4D-Var, AI model을 단계적으로 구현하고 이해하는 것입니다.

---

## Stage 1. Lorenz-63 Forward Model

노트북:

```text
notebooks/01_lorenz63_forward_euler.ipynb
```

목표:

1. Lorenz-63 방정식 정의
2. Euler 시간적분 구현
3. 기준 trajectory 생성
4. `x`, `y`, `z` 시계열 그림 작성
5. Lorenz attractor 그림 작성
6. 초기조건 민감도 확인

예상 source module:

```text
src/lorenz_da/models/lorenz63.py
src/lorenz_da/numerics/euler.py
src/lorenz_da/utils/plotting.py
```

핵심 개념:

- 비선형 동역학계
- explicit Euler method
- chaotic system
- sensitivity to initial condition

---

## Stage 2. Synthetic Observation and 3D-Var

노트북:

```text
notebooks/02_lorenz63_3dvar_euler.ipynb
```

목표:

1. 참값에서 synthetic observation 생성
2. observation operator `H` 정의
3. observation error covariance `R` 정의
4. background error covariance `B` 정의
5. 3D-Var analysis 구현
6. free run과 DA run 비교
7. RMSE 계산 및 시각화

예상 source module:

```text
src/lorenz_da/observations/operator.py
src/lorenz_da/observations/synthetic.py
src/lorenz_da/assimilation/three_dvar.py
src/lorenz_da/utils/diagnostics.py
```

핵심 개념:

- observation operator
- innovation
- background state
- analysis state
- Kalman gain 형태의 update
- background error covariance
- observation error covariance

---

## Stage 3. Tangent Linear Model

노트북:

```text
notebooks/03_lorenz63_tlm_check.ipynb
```

목표:

1. Lorenz-63 방정식의 Jacobian 유도
2. tangent linear model 구현
3. nonlinear perturbation과 linear perturbation 비교
4. finite-difference 방식으로 TLM 검증

예상 source module:

```text
src/lorenz_da/tlm/lorenz63_tlm.py
```

핵심 개념:

- tangent linear approximation
- Jacobian matrix
- perturbation growth
- finite-difference check

---

## Stage 4. Adjoint Model

노트북:

```text
notebooks/04_lorenz63_adjoint_check.ipynb
```

목표:

1. tangent linear model의 adjoint 유도
2. adjoint model 구현
3. inner product test로 adjoint 검증

예상 source module:

```text
src/lorenz_da/adjoint/lorenz63_adjoint.py
```

핵심 개념:

- adjoint operator
- transpose of linearized model
- backward integration
- inner product test

---

## Stage 5. 4D-Var

노트북:

```text
notebooks/05_lorenz63_4dvar_euler.ipynb
```

목표:

1. 4D-Var cost function 정의
2. background term 구현
3. observation term 구현
4. adjoint model을 이용한 gradient 계산
5. gradient descent 또는 scipy optimizer를 이용한 최소화
6. 4D-Var analysis와 truth 비교

예상 source module:

```text
src/lorenz_da/assimilation/four_dvar.py
src/lorenz_da/tlm/lorenz63_tlm.py
src/lorenz_da/adjoint/lorenz63_adjoint.py
```

핵심 개념:

- cost function
- control variable
- assimilation window
- adjoint-based gradient
- optimization
- strong-constraint 4D-Var

---

## Stage 6. AI Surrogate or AI Correction Model

노트북:

```text
notebooks/06_lorenz63_ai_surrogate.ipynb
```

목표:

1. Lorenz-63 trajectory dataset 생성
2. 현재 상태에서 다음 상태를 예측하는 AI model 학습
3. numerical model forecast와 AI forecast 비교
4. AI surrogate model의 장단점 확인
5. model error correction 가능성 실험

예상 source module:

```text
src/lorenz_da/ai/dataset.py
src/lorenz_da/ai/models.py
src/lorenz_da/ai/train.py
```

핵심 개념:

- surrogate model
- one-step prediction
- multi-step rollout
- model error accumulation
- hybrid modeling

---

## Stage 7. Ensemble Kalman Filter (EnKF)

노트북:

```text
notebooks/07_lorenz63_enkf_euler.ipynb
```

목표:

1. 02번 truth, 관측, 3D-Var 결과 재사용
2. 잘못된 초기조건 주변에 초기 앙상블 생성
3. forecast 단계에서 멤버별 Euler 적분
4. perturbed-observation stochastic EnKF analysis 구현
5. EnKF를 free run, 3D-Var와 RMSE로 비교
6. x만 관측할 때 흐름 의존 P^f가 비관측 변수로 정보를 전파하는지 확인
7. 앙상블 크기와 inflation의 영향 확인

예상 source module:

```text
src/lorenz_da/assimilation/enkf.py
```

핵심 개념:

- ensemble
- flow-dependent background error covariance P^f
- perturbed observation (stochastic EnKF)
- ensemble spread와 실제 오차
- multiplicative inflation
- 정적 B(3D-Var)와 P^f(EnKF)의 차이

---

## Stage 8. Lorenz-96 EnKF (localization, inflation)

노트북:

```text
notebooks/08_lorenz96_enkf.ipynb
```

목표:

1. Lorenz-96 모델(F=8, K=40, 순환경계) 구현
2. RK4 시간적분 구현
3. 고차원 공간 확장계에서 EnKF 적용
4. inflation/localization 없을 때 filter divergence 재현
5. localization과 inflation으로 개선

예상 source module:

```text
src/lorenz_da/models/lorenz96.py
src/lorenz_da/numerics/rk4.py
src/lorenz_da/assimilation/localization.py
src/lorenz_da/assimilation/enkf.py
```

핵심 개념:

- 고차원 동역학계
- sampling error
- covariance localization (예: Gaspari-Cohn)
- filter divergence
- inflation의 필요성

---

## Stage 9. Ensemble Optimal Interpolation (EnOI)

노트북:

```text
notebooks/09_lorenz63_enoi_euler.ipynb
```

목표:

1. 긴 free run에서 정적 앙상블(기후학적 변동) 추출
2. 고정된 앙상블 공분산 구성
3. 단일 결정론적 update 수행
4. EnKF(시간 변동 앙상블)와 EnOI(고정 앙상블) 비교

예상 source module:

```text
src/lorenz_da/assimilation/enoi.py
```

핵심 개념:

- climatological ensemble
- stationary covariance
- EnKF와 EnOI의 비용/정확도 trade-off

---

## 개발 원칙

이 프로젝트에서는 다음 원칙을 따른다.

1. 노트북은 설명과 시각화 중심으로 작성한다.
2. 반복해서 사용하는 함수는 `src/lorenz_da/`로 옮긴다.
3. 실험 설정값은 `configs/` 아래에 저장한다.
4. 출력 결과는 `outputs/` 아래에 저장한다.
5. 중요한 수치 계산 함수는 `tests/`에서 검증한다.
6. 처음에는 단순하게 구현하고, 이후 점진적으로 구조화한다.

---

## 현재 우선순위

현재 우선순위는 다음과 같다.

```text
1순위: 01_lorenz63_forward_euler.ipynb 작성        (완료)
2순위: Lorenz-63 forward model을 src module로 분리  (완료)
3순위: 02_lorenz63_3dvar_euler.ipynb 작성          (완료)
4순위: TLM 유도 및 검증                            (완료)
5순위: adjoint 유도 및 검증                        (완료)
6순위: EnKF 구현 및 3D-Var와 비교 (07)             (완료)
7순위: 4D-Var 구현 (05)                            (완료)
8순위: AI surrogate model 실험 (06)               (완료)
9순위: Lorenz-96 EnKF, localization/inflation (08) (완료)
10순위: EnOI 구현 (09)                             (완료)
```

현재 Lorenz DA Lab의 핵심 실습(01~09)은 모두 완성되었다.