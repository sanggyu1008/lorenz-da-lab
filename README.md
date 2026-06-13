# Lorenz DA Lab

Lorenz DA Lab은 Lorenz model을 이용해 수치모델, 자료동화, tangent linear model, adjoint model, 4D-Var, AI model을 단계적으로 공부하기 위한 개인 실습 프로젝트입니다.

이 프로젝트는 실제 해양자료동화 시스템으로 바로 넘어가기 전에, 작은 비선형 동역학계인 Lorenz-63 모델을 이용해 자료동화의 핵심 구조를 직접 구현하고 이해하는 것을 목표로 합니다.

> 이 프로젝트는 4부작 해양 수치모델 실습 시리즈의 하나입니다.
> - **lorenz-da-lab** — 모델에 관측을 *동화(자료동화)* 합니다. ← 현재 저장소
> - [particle-tracking-lab](https://github.com/sanggyu1008/particle-tracking-lab) — 주어진 속도장을 *따라다닙니다(라그랑지안)*.
> - [advection-diffusion-lab](https://github.com/sanggyu1008/advection-diffusion-lab) — 주어진 속도장 위에서 *트레이서를 수송(오일러리안)* 합니다.
> - [shallow-water-lab](https://github.com/sanggyu1008/shallow-water-lab) — 흐름 자체를 *격자 PDE로 생성(전진모델)* 합니다.

---

## 1. 프로젝트 목표

이 저장소의 주요 목표는 다음과 같습니다.

1. Lorenz-63 forward model 구현
2. Euler 및 RK4 시간적분 방법 구현
3. synthetic truth 생성
4. synthetic observation 생성
5. 3D-Var 자료동화 구현
6. tangent linear model 구현 및 검증
7. adjoint model 구현 및 검증
8. 4D-Var cost function과 gradient 계산 구현
9. AI surrogate model 또는 AI correction model 실험
10. Ensemble Kalman Filter(EnKF) 구현 및 3D-Var와 비교
11. Ensemble Optimal Interpolation(EnOI) 구현
12. Lorenz-96 EnKF에서 localization과 inflation 실험

---

## 2. 학습 순서

전체 실습은 다음 순서로 진행합니다.

```text
01. Lorenz-63 forward model
02. Synthetic observation과 3D-Var
03. Tangent linear model
04. Adjoint model
05. 4D-Var
06. AI surrogate / AI correction model
07. Ensemble Kalman Filter (EnKF)
08. Ensemble Optimal Interpolation (EnOI)
09. Lorenz-96 EnKF (localization, inflation)
```

> 변분(variational) 계열은 02 → 03 → 04 → 05로, 앙상블 계열은 02 → 07 → 08 → 09로 이어집니다.

추천 노트북 구성은 다음과 같습니다.

```text
notebooks/
├── 01_lorenz63_forward_euler.ipynb
├── 02_lorenz63_3dvar_euler.ipynb
├── 03_lorenz63_tlm_check.ipynb
├── 04_lorenz63_adjoint_check.ipynb
├── 05_lorenz63_4dvar_euler.ipynb
├── 06_lorenz63_ai_surrogate.ipynb
├── 07_lorenz63_enkf_euler.ipynb
├── 08_lorenz63_enoi_euler.ipynb
└── 09_lorenz96_enkf.ipynb
```

---

## 3. 디렉토리 구조

```text
lorenz-da-lab/
├── configs/
│   ├── model/
│   ├── assimilation/
│   └── ai/
│
├── data/
│   ├── raw/
│   ├── synthetic/
│   └── processed/
│
├── docs/
│
├── environment/
│   ├── environment.yml
│   └── requirements.txt
│
├── notebooks/
│
├── outputs/
│   ├── figures/
│   ├── trajectories/
│   ├── assimilation/
│   ├── checkpoints/
│   └── logs/
│
├── src/
│   └── lorenz_da/
│       ├── models/
│       ├── numerics/
│       ├── observations/
│       ├── assimilation/
│       ├── tlm/
│       ├── adjoint/
│       ├── ai/
│       └── utils/
│
└── tests/
```

---

## 4. 환경 설정

시리즈 4개 프로젝트는 모두 같은 conda 환경 `numlab` 을 공유합니다.

### 4.1 Conda 환경 생성

```bash
conda env create -f environment/environment.yml
conda activate numlab
```

Jupyter Notebook에서 사용할 수 있도록 kernel을 등록합니다.

```bash
python -m ipykernel install --user --name numlab --display-name "Python (numlab)"
```

### 4.2 pip 기반 설치

Conda 대신 Python virtual environment를 사용할 수도 있습니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements.txt
pip install -e .
```

---

## 5. import 방식

이 프로젝트는 `src/` 레이아웃을 사용합니다. 두 가지 방법으로 패키지를 import할 수 있습니다.

1. **editable 설치** (`pip install -e .`) 후 어디서나:

   ```python
   from lorenz_da.models.lorenz63 import lorenz63_rhs
   from lorenz_da.numerics.euler import euler_step
   ```

2. **설치 없이** 노트북 안에서 `src/`를 `sys.path`에 직접 추가 (각 노트북 상단 셀이 자동 처리). 프로젝트 폴더를 옮겨도 경로가 깨지지 않도록 `find_project_root()`로 루트를 찾습니다.

---

## 6. 실습 노트북 (전체 완성)

9단계 노트북이 모두 작성·실행 완료되었습니다. 순서대로 따라가면 됩니다.

```text
notebooks/
├── 01_lorenz63_forward_euler.ipynb   # Lorenz-63 forward model, Euler 적분, attractor
├── 02_lorenz63_3dvar_euler.ipynb     # synthetic 관측 + 3D-Var 자료동화
├── 03_lorenz63_tlm_check.ipynb       # tangent linear model 구현·검증
├── 04_lorenz63_adjoint_check.ipynb   # adjoint model 구현·gradient 검증
├── 05_lorenz63_4dvar_euler.ipynb     # 4D-Var (strong-constraint, L-BFGS)
├── 06_lorenz63_ai_surrogate.ipynb    # AI surrogate (numpy MLP + Adam)
├── 07_lorenz63_enkf_euler.ipynb      # Ensemble Kalman Filter (L63, stochastic)
├── 08_lorenz63_enoi_euler.ipynb      # Ensemble Optimal Interpolation (EnOI)
└── 09_lorenz96_enkf.ipynb            # Lorenz-96 EnKF (localization, inflation)
```

첫 번째 노트북(`01`)의 목표는 다음과 같습니다.

1. Lorenz-63 방정식 이해
2. Lorenz-63 RHS 함수 구현
3. Euler 방법으로 시간적분
4. `x`, `y`, `z` 시계열 시각화
5. Lorenz attractor 시각화
6. 초기조건 민감도 확인

```bash
PYTHONPATH=src python -m pytest         # 29 passed
```

---

## 7. 출력물 관리 규칙

실습 중 생성되는 결과물은 `outputs/` 아래에 저장합니다.

```text
outputs/
├── figures/        # 그림 파일
├── trajectories/   # Lorenz trajectory 파일
├── assimilation/   # 자료동화 실험 결과
├── checkpoints/    # AI model checkpoint
└── logs/           # 실험 로그
```

큰 출력 파일은 Git에 올리지 않습니다. 단, 디렉토리 구조 유지를 위한 `.gitkeep`은 남깁니다.

---

## 8. 개발 원칙

1. 노트북은 개념 설명과 시각화 중심으로 작성합니다.
2. 반복해서 사용하는 함수는 `src/lorenz_da/` 아래로 옮깁니다.
3. 중요한 수치 계산 함수는 `tests/`에서 검증합니다.
4. 실험 설정값은 가능하면 `configs/` 아래에 저장합니다.
5. 출력물은 `outputs/` 아래에 저장합니다.
6. 임시 실험 파일은 `notebooks/_scratch/`에 둡니다.

---

## 9. 향후 계획

자세한 학습 계획은 다음 문서를 참고합니다.

```text
docs/roadmap.md
```
