from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed python + numpy and return a `np.random.Generator`.

    Why this exists:
    - `np.random.seed()` does *not* affect `default_rng()` instances.
    - We want deterministic pipelines by default.

    Returns
    -------
    np.random.Generator
        A generator you can pass around.
    """
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, used for lightweight provenance."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=True)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dataclass_to_dict(dc: Any) -> Dict[str, Any]:
    """Safe serialization for dataclasses used in configs."""
    return asdict(dc)
