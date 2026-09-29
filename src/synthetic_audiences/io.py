from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from synthetic_audiences.schemas import Artifact, Question
from synthetic_audiences.utils import ensure_parent


def read_agents_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_questions_csv(path: str | Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    questions: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        options = row.get("options")
        if isinstance(options, str) and options.strip():
            row["options"] = [part.strip() for part in options.split(";")]
        else:
            row["options"] = None
        questions.append(Question(**row).model_dump(by_alias=True))
    return questions


def read_artifact_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return Artifact(**json.load(f)).model_dump()


def append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = ensure_parent(path)
    with out.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: Any) -> None:
    out = ensure_parent(path)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
