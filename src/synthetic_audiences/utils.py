from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(float(v) for v in weights.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return {k: float(v) / total for k, v in weights.items()}


def weighted_choice(rng: random.Random, weights: Mapping[str, float]) -> str:
    norm = normalize_weights(weights)
    labels = list(norm.keys())
    probs = list(norm.values())
    return rng.choices(labels, weights=probs, k=1)[0]


def deterministic_id(prefix: str, payload: dict[str, Any], digits: int = 10) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:digits]
    return f"{prefix}_{digest}"


def ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def stable_float_0_1(*parts: object) -> float:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:12]
    return int(digest, 16) / float(0xFFFFFFFFFFFF)
