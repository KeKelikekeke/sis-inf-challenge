"""Parsing and loading for challenge problem instance files."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from sisinf.paths import get_problem_file
from sisinf.problem_meta import get_problem_spec
from sisinf.types import Instance


def _strip_simple_comments(text: str) -> str:
    """Remove whole-line and trailing simple comments from data text."""

    cleaned: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            cleaned.append("")
            continue
        cleaned.append(re.split(r"\s(?:#|//)", line, maxsplit=1)[0])
    return "\n".join(cleaned)


def _literal_shape(value: Any) -> str:
    """Return a best-effort shape string for diagnostics."""

    try:
        return str(np.asarray(value, dtype=object).shape)
    except Exception:
        return "<unknown>"


def _parse_literal(value_text: str, path: Path, field: str) -> Any:
    """Parse one extracted literal safely with Python or JSON parsers."""

    errors: list[str] = []
    for parser_name, parser in (("ast.literal_eval", ast.literal_eval), ("json.loads", json.loads)):
        try:
            return parser(value_text)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{parser_name}: {exc}")
    preview = value_text[:160].replace("\n", " ")
    raise ValueError(f"Failed parsing {field} in {path}: {'; '.join(errors)}. Preview: {preview!r}")


def _extract_balanced_value(text: str, start_pos: int, path: Path, field: str) -> str:
    """Extract a bracketed literal starting after an assignment."""

    pos = start_pos
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text) or text[pos] not in "[{(":
        snippet = text[start_pos : start_pos + 80].replace("\n", " ")
        raise ValueError(f"Expected bracketed literal for {field} in {path}; got {snippet!r}")

    pairs = {"[": "]", "{": "}", "(": ")"}
    stack = [pairs[text[pos]]]
    in_string: str | None = None
    escaped = False

    for idx in range(pos + 1, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            continue

        if char in {"'", '"'}:
            in_string = char
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return text[pos : idx + 1]

    raise ValueError(f"Unclosed bracketed literal for {field} in {path}, starting near offset {pos}")


def _extract_assignment(text: str, key: str, path: Path) -> Any | None:
    """Extract a top-level ``key = literal`` or ``key: literal`` value."""

    pattern = re.compile(rf"(?m)(?:^|[\s{{,])['\"]?{re.escape(key)}['\"]?\s*[:=]\s*")
    match = pattern.search(text)
    if match is None:
        return None
    return _parse_literal(_extract_balanced_value(text, match.end(), path, key), path, key)


def _parse_mapping(text: str, path: Path) -> dict[str, Any] | None:
    """Try parsing the whole file as a mapping object."""

    stripped = text.strip()
    if not stripped:
        raise ValueError(f"Problem file is empty: {path}")
    if not stripped.startswith("{"):
        return None

    errors: list[str] = []
    for parser_name, parser in (("json.loads", json.loads), ("ast.literal_eval", ast.literal_eval)):
        try:
            value = parser(stripped)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{parser_name}: {exc}")
            continue
        if not isinstance(value, dict):
            raise ValueError(f"Parsed whole file {path} as {type(value).__name__}; expected mapping")
        return value
    raise ValueError(f"Failed parsing whole-file mapping in {path}: {'; '.join(errors)}")


def _read_problem_literals(path: Path) -> tuple[Any, Any | None]:
    """Read raw ``A`` and optional ``t`` literals from a problem file."""

    text = _strip_simple_comments(path.read_text(encoding="utf-8"))
    mapping = _parse_mapping(text, path)
    if mapping is not None:
        if "A" not in mapping:
            raise ValueError(f"Missing key A in mapping parsed from {path}; keys={sorted(mapping)}")
        return mapping["A"], mapping.get("t")

    raw_A = _extract_assignment(text, "A", path)
    if raw_A is None:
        raise ValueError(f"Missing A assignment in {path}; expected syntax like 'A = [[...], ...]'")
    return raw_A, _extract_assignment(text, "t", path)


def _normalize_A_from_column_format(raw_A: Any, n: int, m: int) -> np.ndarray:
    """Convert raw column-form ``A`` with shape ``(m, n)`` into shape ``(n, m)``."""

    try:
        raw_arr = np.asarray(raw_A, dtype=np.int64)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "Failed converting A to int64 array: "
            f"type={type(raw_A).__name__}, inferred_shape={_literal_shape(raw_A)}"
        ) from exc

    if raw_arr.ndim != 2:
        raise ValueError(f"Raw A must be two-dimensional; got shape {raw_arr.shape}")
    if raw_arr.shape != (m, n):
        raise ValueError(f"Raw A has shape {raw_arr.shape}; expected column-format shape ({m}, {n})")
    return raw_arr.T.copy()


def _normalize_t(raw_t: Any, n: int, path: Path) -> np.ndarray:
    """Convert raw ``t`` to an int64 vector of shape ``(n,)``."""

    try:
        t_arr = np.asarray(raw_t, dtype=np.int64).reshape(-1)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Failed converting t in {path} to int64 vector: "
            f"type={type(raw_t).__name__}, inferred_shape={_literal_shape(raw_t)}"
        ) from exc
    if t_arr.shape != (n,):
        raise ValueError(f"t in {path} has shape {t_arr.shape}; expected ({n},)")
    return t_arr


def parse_problem_file(path: Path, index: int) -> Instance:
    """Parse one problem file into a normalized :class:`Instance`."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Problem file not found for index {index}: {path}")

    spec = get_problem_spec(index)
    n = int(spec["n"])
    m = int(spec["m"])
    raw_A, raw_t = _read_problem_literals(path)

    try:
        A = _normalize_A_from_column_format(raw_A, n=n, m=m)
    except ValueError as exc:
        raise ValueError(
            f"Failed normalizing A from {path}: type={type(raw_A).__name__}, "
            f"inferred_shape={_literal_shape(raw_A)}. {exc}"
        ) from exc

    homogeneous = bool(spec["homogeneous"])
    if raw_t is None:
        if not homogeneous:
            raise ValueError(f"Missing t in inhomogeneous problem{index} file: {path}")
        t = np.zeros(n, dtype=np.int64)
    else:
        t = _normalize_t(raw_t, n=n, path=path)

    return Instance(
        name=f"problem{index}",
        index=index,
        n=n,
        m=m,
        q=int(spec["q"]),
        gamma=int(spec["gamma"]),
        A=A,
        t=t,
        require_l2_ge_q=bool(spec["require_l2_ge_q"]),
        homogeneous=homogeneous,
        source_path=path,
    )


def load_problem(index: int) -> Instance:
    """Load a challenge instance by index from the repository data directory."""

    return parse_problem_file(get_problem_file(index), index=index)
