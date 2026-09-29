import json
import time
import uuid
from pathlib import Path
import pandas as pd

from src.synthetic_audiences.agents.context import safe_get
from src.synthetic_audiences.agents.chain import build_agent_chain, run_agent_question
from src.synthetic_audiences.settings import (
    SOCIO_DEMOGRAPHIC_COLUMNS,
    VALUES_ATTITUDES_PERCEPTIONS_COLUMNS,
    DEFAULT_REPEATS,
    DEFAULT_MODEL,
)


def run_questionnaire(
    df_agents: pd.DataFrame,
    questionnaire: list[dict],
    artifact: dict,
    output_dir: str | Path,
    limit_agents: int | None = None,
    repeats: int = DEFAULT_REPEATS,
    model: str = DEFAULT_MODEL,
    sleep_seconds: float = 0.2,
) -> tuple[pd.DataFrame, dict]:
    """
    Aplica um questionário para uma população de agentes.

    Default: repeats=1.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"run_langchain_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    responses_jsonl = output_dir / f"{run_id}_responses.jsonl"
    responses_csv = output_dir / f"{run_id}_responses.csv"

    df_run_agents = df_agents.copy() if limit_agents is None else df_agents.head(limit_agents).copy()

    chain = build_agent_chain()
    records = []
    errors = []
    call_n = 0

    with responses_jsonl.open("w", encoding="utf-8") as f:
        for _, agent_row in df_run_agents.iterrows():
            for question in questionnaire:
                for repeat_id in range(repeats):
                    call_n += 1

                    try:
                        answer = run_agent_question(
                            agent_row=agent_row,
                            question=question,
                            artifact=artifact,
                            chain=chain,
                        )

                        variable_reading = answer.get("agent_variable_reading", {})

                        record = {
                            "run_id": run_id,
                            "model": model,
                            "artifact_id": artifact["artifact_id"],
                            "artifact_type": artifact["artifact_type"],
                            "artifact_title": artifact["title"],
                            "artifact_tipo_de_fontes_de_informação": artifact["tipo_de_fontes_de_informação"],
                            "id_persona": safe_get(agent_row, "id_persona"),
                            "Persona": safe_get(agent_row, "Persona"),

                            **{col: safe_get(agent_row, col) for col in SOCIO_DEMOGRAPHIC_COLUMNS},
                            **{col: safe_get(agent_row, col) for col in VALUES_ATTITUDES_PERCEPTIONS_COLUMNS},

                            "question_id": question["question_id"],
                            "construct": question["construct"],
                            "question": question["question"],
                            "repeat_id": repeat_id,

                            "answer_value": answer.get("answer_value"),
                            "answer_label": answer.get("answer_label"),
                            "rationale": answer.get("rationale"),
                            "confidence": answer.get("confidence"),
                            "fonte_familiar": answer.get("fonte_familiar"),
                            "source_familiarity_effect": answer.get("source_familiarity_effect"),

                            "persona_summary": variable_reading.get("persona_summary"),
                            "socio_demographic_summary": variable_reading.get("socio_demographic_summary"),
                            "values_attitudes_perceptions_summary": variable_reading.get("values_attitudes_perceptions_summary"),

                            "raw_answer": json.dumps(answer, ensure_ascii=False),
                        }

                        records.append(record)
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    except Exception as e:
                        errors.append(
                            {
                                "call_n": call_n,
                                "id_persona": safe_get(agent_row, "id_persona"),
                                "question_id": question["question_id"],
                                "repeat_id": repeat_id,
                                "error": repr(e),
                            }
                        )

                    time.sleep(sleep_seconds)

    df_responses = pd.DataFrame(records)
    df_responses.to_csv(responses_csv, index=False, sep=";", encoding="utf-8-sig")

    metadata = {
        "run_id": run_id,
        "responses_jsonl": str(responses_jsonl),
        "responses_csv": str(responses_csv),
        "errors": errors,
        "n_responses": len(df_responses),
        "n_errors": len(errors),
    }

    return df_responses, metadata
