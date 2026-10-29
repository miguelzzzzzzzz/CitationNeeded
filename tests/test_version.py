"""The package version must match the version in pyproject.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

import rag_engine

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_matches_pyproject() -> None:
    with PYPROJECT.open("rb") as fh:
        project_version = tomllib.load(fh)["project"]["version"]
    assert rag_engine.__version__ == project_version
