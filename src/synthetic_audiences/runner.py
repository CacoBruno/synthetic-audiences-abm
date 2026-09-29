from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from synthetic_audiences.io import append_jsonl
from synthetic_audiences.llm import OpenAIJsonClient, mock_agent_response
from synthetic_audiences.prompts import build_prompt_agent
from synthetic_audiences.schemas import AgentRunResponse
from synthetic_audiences.utils import ensure_parent


def _coerce_response(
    raw: dict[str, Any],
    agent_id: str,
    artifact_id: str,
    question: dict[str, Any],
    iteration: int,
) -> dict[str, Any]:
    response_type = question.get("response_type", "likert_1_5")
    answer = raw.get("answer")
    score = raw.get("score_numeric")
    if response_type == "likert_1_5":
        try:
            answer = int(float(answer))
            answer = max(1, min(5, answer))
            score = float(answer)
        except Exception:
            answer = 3
            score = 3.0
    else:
        if score in {"", "null", None}:
            score = None
        elif score is not None:
            try:
                score = float(score)
            except Exception:
                score = None

    confidence = raw.get("confidence", 0.5)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except Exception:
        confidence = 0.5

    payload = {
        "agent_id": agent_id,
        "artifact_id": artifact_id,
        "question_id": question["question_id"],
        "construct": question["construct"],
        "iteration": iteration,
        "response_type": response_type,
        "answer": answer,
        "score_numeric": score,
        "rationale": str(raw.get("rationale", ""))[:1200],
        "confidence": confidence,
        "raw_response": raw,
    }
    return AgentRunResponse(**payload).model_dump(by_alias=True)


def apply_questionnaire(
    agents: pd.DataFrame,
    questions: list[dict[str, Any]],
    artifact: dict[str, Any],
    repeats: int = 3,
    output_jsonl: str | Path | None = None,
    use_mock: bool = False,
    model: str | None = None,
    task: str | None = None,
    reasoning_mode: str = "thinking",
    limit_agents: int | None = None,
) -> pd.DataFrame:
    """Apply questionnaire questions to each agent for a given artifact.

    Returns one row per agent x question x iteration and optionally appends raw rows
    to JSONL. Use repeats=3 or 5 to estimate mean/stdev for Likert outputs.
    """
    if repeats <= 0:
        raise ValueError("repeats must be > 0")
    work_agents = agents.head(limit_agents) if limit_agents else agents
    client = None if use_mock else OpenAIJsonClient(model=model)
    rows: list[dict[str, Any]] = []
    total = len(work_agents) * len(questions) * repeats

    with tqdm(total=total, desc="Applying agents") as pbar:
        for _, agent in work_agents.iterrows():
            agent_id = str(agent["id_persona"])
            for question in questions:
                for iteration in range(repeats):
                    if use_mock:
                        raw = mock_agent_response(agent_id, question, artifact, iteration)
                    else:
                        messages = build_prompt_agent(
                            agent_profile=agent,
                            question=question,
                            artifact=artifact,
                            task=task,
                            reasoning_mode=reasoning_mode,
                        )
                        raw = client.complete_json(messages)  # type: ignore[union-attr]
                    rows.append(
                        _coerce_response(
                            raw=raw,
                            agent_id=agent_id,
                            artifact_id=artifact["artifact_id"],
                            question=question,
                            iteration=iteration,
                        )
                    )
                    if output_jsonl and len(rows) >= 50:
                        append_jsonl(output_jsonl, rows)
                        rows.clear()
                    pbar.update(1)

    if output_jsonl and rows:
        append_jsonl(output_jsonl, rows)
    if output_jsonl:
        return pd.read_json(output_jsonl, lines=True)
    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame, output_csv: str | Path | None = None) -> pd.DataFrame:
    """Aggregate mean/std by artifact, construct and question for numeric outputs."""
    numeric = results.dropna(subset=["score_numeric"]).copy()
    if numeric.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            numeric.groupby(["artifact_id", "construct", "question_id"], as_index=False)
            .agg(
                mean_score=("score_numeric", "mean"),
                std_score=("score_numeric", "std"),
                n_responses=("score_numeric", "count"),
                mean_confidence=("confidence", "mean"),
            )
            .sort_values(["artifact_id", "construct", "question_id"])
        )
    if output_csv:
        out = ensure_parent(output_csv)
        summary.to_csv(out, index=False, encoding="utf-8-sig")
    return summary
