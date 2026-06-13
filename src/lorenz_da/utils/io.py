"""
경로/입출력 유틸리티.

노트북과 script가 공통으로 쓰는 경로 처리를 한곳에 모은다.

핵심 함수
---------
- find_project_root : cwd에서 위로 올라가며 프로젝트 루트를 찾는다.
- ensure_src_on_path : src/ 를 sys.path에 추가한다(editable 설치에 의존하지 않게).
- get_output_dirs : outputs/ 하위 표준 디렉토리들을 만들어 dict로 반환한다.

노트북에는 __file__ 이 없으므로, 작업 디렉토리(cwd)에서 pyproject.toml 또는 .git이
있는 디렉토리를 프로젝트 루트로 본다. 프로젝트 폴더를 다른 경로로 옮겨도 동작한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_MARKERS = ("pyproject.toml", ".git")

# outputs/ 아래 표준 하위 디렉토리
OUTPUT_SUBDIRS = {
    "figures": ("outputs", "figures"),
    "trajectories": ("outputs", "trajectories"),
    "assimilation": ("outputs", "assimilation"),
    "checkpoints": ("outputs", "checkpoints"),
    "logs": ("outputs", "logs"),
}


def find_project_root(
    start: str | Path | None = None,
    markers: tuple[str, ...] = PROJECT_MARKERS,
) -> Path:
    """
    프로젝트 루트 디렉토리를 찾는다.

    start(기본값: 현재 작업 디렉토리)에서 부모 디렉토리를 따라 올라가며,
    markers 중 하나라도 포함하는 첫 디렉토리를 루트로 본다.

    Parameters
    ----------
    start : str or Path or None
        탐색을 시작할 경로. None이면 Path.cwd()를 사용한다.
    markers : tuple of str
        루트 판별에 쓸 파일/디렉토리 이름들.

    Returns
    -------
    root : Path
        프로젝트 루트. 마커를 못 찾으면 notebooks/ 안에서 실행한 경우로 보고
        그 부모를, 아니면 start 자체를 반환한다.
    """
    start_path = Path.cwd() if start is None else Path(start)
    start_path = start_path.resolve()

    for path in (start_path, *start_path.parents):
        if any((path / marker).exists() for marker in markers):
            return path

    return start_path.parent if start_path.name == "notebooks" else start_path


def ensure_src_on_path(root: str | Path | None = None) -> Path:
    """
    프로젝트의 src/ 디렉토리를 sys.path에 추가한다.

    editable 설치(`pip install -e .`)의 위치에 의존하지 않고 lorenz_da를
    import할 수 있게 한다. 프로젝트를 다른 경로로 옮겨도 동작한다.

    Parameters
    ----------
    root : str or Path or None
        프로젝트 루트. None이면 find_project_root()로 찾는다.

    Returns
    -------
    root : Path
        사용된 프로젝트 루트.
    """
    root_path = find_project_root() if root is None else Path(root)
    src = root_path / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root_path


def get_output_dirs(
    root: str | Path | None = None,
    create: bool = True,
) -> dict[str, Path]:
    """
    outputs/ 아래 표준 디렉토리들을 dict로 반환한다.

    Parameters
    ----------
    root : str or Path or None
        프로젝트 루트. None이면 find_project_root()로 찾는다.
    create : bool
        True이면 디렉토리를 만들어 둔다.

    Returns
    -------
    dirs : dict[str, Path]
        {"figures", "trajectories", "assimilation", "checkpoints", "logs"} 경로.
    """
    root_path = find_project_root() if root is None else Path(root)
    dirs = {name: root_path.joinpath(*parts) for name, parts in OUTPUT_SUBDIRS.items()}

    if create:
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    return dirs
