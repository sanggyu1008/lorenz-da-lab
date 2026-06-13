"""
간단한 multilayer perceptron (MLP)을 numpy로 직접 구현한다.

외부 딥러닝 라이브러리(torch 등) 없이, 신경망 surrogate model이 어떻게
forward 계산을 하는지 명시적으로 보여 주기 위한 교육용 구현이다.

은닉층은 tanh 활성함수를, 출력층은 선형(활성함수 없음)을 사용한다.
"""

from __future__ import annotations

import numpy as np


class MLP:
    """
    tanh 은닉층 + 선형 출력층 MLP.

    Parameters
    ----------
    layer_sizes : sequence of int
        각 층의 노드 수. 예: [3, 32, 32, 3]은 입력 3, 은닉 32-32, 출력 3.
    rng : np.random.Generator or None
        가중치 초기화에 쓸 난수 생성기.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        rng: np.random.Generator | None = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes는 최소 입력, 출력 2개 이상이어야 한다.")

        if rng is None:
            rng = np.random.default_rng()

        self.layer_sizes = list(layer_sizes)
        self.params: list[list[np.ndarray]] = []

        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He 초기화 (tanh에도 무난하게 동작)
            W = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
            b = np.zeros(n_out, dtype=float)
            self.params.append([W, b])

    def forward(
        self,
        X: np.ndarray,
        return_cache: bool = False,
    ):
        """
        forward 계산을 수행한다.

        Parameters
        ----------
        X : np.ndarray, shape (batch, n_in)
            입력 배치.
        return_cache : bool
            True이면 backprop에 필요한 중간값(cache)을 함께 반환한다.

        Returns
        -------
        output : np.ndarray, shape (batch, n_out)
        cache : list (return_cache=True일 때만)
        """
        X = np.asarray(X, dtype=float)
        a = X
        caches = []
        n_layers = len(self.params)

        for i, (W, b) in enumerate(self.params):
            z = a @ W + b
            caches.append((a, z))
            if i < n_layers - 1:
                a = np.tanh(z)
            else:
                a = z  # 선형 출력

        if return_cache:
            return a, caches
        return a

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def get_flat_params(self) -> np.ndarray:
        """모든 가중치를 1차원 벡터로 펼쳐 반환한다(검증/저장용)."""
        return np.concatenate([p.ravel() for W, b in self.params for p in (W, b)])
