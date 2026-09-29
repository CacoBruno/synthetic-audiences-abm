from __future__ import annotations

import argparse
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_audiences.experiments.credit_narratives import (
    balanced_limit_agents,
    build_experiment_messages,
    build_response_schema,
    mock_experiment_response,
    questionnaire_for_condition,
    response_to_long_rows,
)
from synthetic_audiences.llm import OpenAIJsonClient


HERE = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_stimuli(stimuli_config: dict, themes: set[str] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for theme_id, theme_cfg in stimuli_config["themes"].items():
        if themes and theme_id not in themes:
            continue
        for stimulus in theme_cfg["stimuli"]:
            row = dict(stimulus)
            row["theme"] = theme_id
            out.append(row)
    return out


def coerce_response(
    response: dict[str, Any],
    items: list[dict[str, Any]],
    qualitative: dict[str, Any],
) -> dict[str, Any]:
    out = dict(response)
    for item in items:
        value = out.get(item["id"], 4)
        try:
            value = int(round(float(value)))
        except Exception:
            value = 4
        out[item["id"]] = max(
            int(item.get("scale_min", 1)),
            min(int(item.get("scale_max", 7)), value),
        )

    categories = qualitative["categories"]
    if out.get("interpretation_category") not in categories:
        out["interpretation_category"] = categories[-1]

    out["interpretation_open"] = str(out.get("interpretation_open", ""))[:2000]
    out["critique_open"] = str(out.get("critique_open", ""))[:2000]
    out["overall_rationale"] = str(out.get("overall_rationale", ""))[:1200]

    try:
        confidence = float(out.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    out["confidence"] = max(0.0, min(1.0, confidence))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa o teste de narrativas de crédito do Itaú."
    )
    parser.add_argument("--agents", type=Path, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--reasoning-effort", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--limit-agents", type=int, default=None)
    parser.add_argument("--theme", action="append", default=None)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    config = load_yaml(HERE / "config.yml")
    stimuli_config = load_yaml(HERE / "stimuli.yml")
    questionnaire = load_yaml(HERE / "questionnaire.yml")
    exp = config["experiment"]

    agents_path = args.agents or (HERE / "outputs" / "agents_itau_credit.csv")
    if not agents_path.exists():
        raise FileNotFoundError(
            f"Agentes experimentais não encontrados: {agents_path}. "
            "Rode primeiro build_agents.py."
        )

    agents = pd.read_csv(agents_path)
    agents = balanced_limit_agents(
        agents,
        args.limit_agents,
        seed=int(exp["seed"]),
    )

    model = args.model or exp["model"]
    reasoning_effort = args.reasoning_effort or exp["reasoning_effort"]
    repeats = args.repeats or int(exp["repeats"])
    max_workers = args.max_workers or int(exp["max_workers"])
    themes = set(args.theme) if args.theme else None
    stimuli = flatten_stimuli(stimuli_config, themes=themes)

    run_id = (
        f"{exp['id']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex[:6]}"
    )
    output_root = ROOT / config["paths"]["outputs_dir"]
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    client = None
    if not args.mock:
        client = OpenAIJsonClient(
            model=model,
            reasoning_effort=reasoning_effort,
        )

    tasks: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for _, agent in agents.iterrows():
        agent_dict = agent.to_dict()
        for stimulus in stimuli:
            for repeat_id in range(repeats):
                tasks.append((agent_dict, stimulus, repeat_id))

    def execute(task):
        agent_dict, stimulus, repeat_id = task
        agent_row = pd.Series(agent_dict)
        items, qualitative = questionnaire_for_condition(
            questionnaire,
            stimulus["theme"],
            stimulus["condition"],
        )

        if args.mock:
            response = mock_experiment_response(
                str(agent_row["id_persona"]),
                stimulus,
                items,
                qualitative,
                repeat_id,
            )
        else:
            messages = build_experiment_messages(
                agent_row,
                stimulus,
                items,
                qualitative,
            )
            schema = build_response_schema(items, qualitative)
            response = client.complete_json_schema(
                messages,
                schema_name="itau_credit_narrative_response",
                schema=schema,
                strict=True,
            )

        response = coerce_response(response, items, qualitative)
        long_rows = response_to_long_rows(
            agent_row,
            stimulus,
            items,
            response,
            run_id=run_id,
            model=model,
            iteration=repeat_id,
            is_mock=args.mock,
        )

        wide = {
            "run_id": run_id,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "is_mock": args.mock,
            "theme": stimulus["theme"],
            "stimulus_id": stimulus["id"],
            "condition": stimulus["condition"],
            "repeat_id": repeat_id,
            "id_persona": str(agent_row["id_persona"]),
            "id_persona_base": str(agent_row.get("id_persona_base", "")),
            "perfil_credito": str(agent_row.get("perfil_credito", "")),
            "perfil_credito_label": str(agent_row.get("perfil_credito_label", "")),
            "perfil_credito_subtipo": str(agent_row.get("perfil_credito_subtipo", "")),
            **{item["id"]: response[item["id"]] for item in items},
            "interpretation_category": response["interpretation_category"],
            "interpretation_open": response["interpretation_open"],
            "critique_open": response["critique_open"],
            "overall_rationale": response["overall_rationale"],
            "confidence": response["confidence"],
        }
        return wide, long_rows

    wide_records: list[dict[str, Any]] = []
    long_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(execute, task): task
            for task in tasks
        }
        for future in tqdm(
            as_completed(future_map),
            total=len(future_map),
            desc="Narrative experiment",
        ):
            agent_dict, stimulus, repeat_id = future_map[future]
            try:
                wide, long_rows = future.result()
                wide_records.append(wide)
                long_records.extend(long_rows)
            except Exception as exc:
                errors.append(
                    {
                        "id_persona": str(agent_dict.get("id_persona", "")),
                        "perfil_credito": str(agent_dict.get("perfil_credito", "")),
                        "stimulus_id": stimulus["id"],
                        "theme": stimulus["theme"],
                        "condition": stimulus["condition"],
                        "repeat_id": repeat_id,
                        "error": repr(exc),
                    }
                )

    wide_df = pd.DataFrame(wide_records)
    long_df = pd.DataFrame(long_records)

    wide_df.to_json(
        run_dir / "responses_wide.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    long_df.to_csv(
        run_dir / "responses_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (run_dir / "errors.json").open("w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    metadata = {
        "run_id": run_id,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "is_mock": args.mock,
        "n_agents": int(agents["id_persona"].nunique()),
        "n_stimuli": len(stimuli),
        "repeats": repeats,
        "planned_calls": len(tasks),
        "successful_calls": len(wide_records),
        "errors": len(errors),
        "max_workers": max_workers,
        "profile_counts": {
            str(k): int(v)
            for k, v in agents["perfil_credito"].value_counts().sort_index().items()
        },
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Resultados: {run_dir}")


if __name__ == "__main__":
    main()
