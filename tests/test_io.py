"""utils/io.py 경로 헬퍼 검증."""

from __future__ import annotations

from pathlib import Path

from lorenz_da.utils.io import find_project_root, get_output_dirs


def test_find_project_root_locates_pyproject():
    # 이 테스트 파일 위치에서 시작해도 lorenz-da-lab 루트를 찾아야 한다.
    root = find_project_root(start=Path(__file__).parent)
    assert (root / "pyproject.toml").exists()


def test_find_project_root_from_notebooks_dir():
    root = find_project_root(start=Path(__file__).parent)
    notebooks = root / "notebooks"
    if notebooks.exists():
        # notebooks/ 안에서 실행해도 같은 루트를 찾는다.
        assert find_project_root(start=notebooks) == root


def test_get_output_dirs_keys_and_paths(tmp_path):
    dirs = get_output_dirs(root=tmp_path, create=True)
    assert set(dirs) == {
        "figures",
        "trajectories",
        "assimilation",
        "checkpoints",
        "logs",
    }
    for name, path in dirs.items():
        assert path.exists()
        assert path == tmp_path / "outputs" / name
