"""Repository-relative path helpers for challenge data files."""

from __future__ import annotations

from pathlib import Path


def get_repo_root() -> Path:
    """Locate the repository root by searching for ``data/raw_data``."""

    candidates: list[Path] = []
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        candidates.extend([start, *start.parents])

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "data" / "raw_data").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate repository root containing data/raw_data. "
        f"Searched from module path {Path(__file__).resolve()} and cwd {Path.cwd().resolve()}."
    )


def get_raw_data_dir() -> Path:
    """Return the raw data directory path."""

    path = get_repo_root() / "data" / "raw_data"
    if not path.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {path}")
    return path


def get_problem_dir() -> Path:
    """Return the directory containing ``problem{i}.txt`` files."""

    path = get_raw_data_dir() / "sis_inf_problems"
    if not path.is_dir():
        raise FileNotFoundError(f"SIS infinity problem directory not found: {path}")
    return path


def get_problem_file(index: int) -> Path:
    """Return the path for a challenge instance text file."""

    path = get_problem_dir() / f"problem{index}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Problem file for index {index} not found: {path}")
    return path


def get_problem_pdf() -> Path:
    """Return the original problem statement PDF path."""

    path = get_raw_data_dir() / "2026密码数学挑战赛-赛题一.pdf"
    if not path.is_file():
        raise FileNotFoundError(f"Problem statement PDF not found: {path}")
    return path
