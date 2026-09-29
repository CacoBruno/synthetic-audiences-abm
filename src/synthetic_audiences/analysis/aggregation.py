from pathlib import Path
import pandas as pd
from src.synthetic_audiences.settings import DEFAULT_GROUP_VARIABLES


def aggregate_by_construct(
    df_responses: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    df = df_responses.copy()
    df["answer_value"] = pd.to_numeric(df["answer_value"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    out = (
        df
        .groupby(["artifact_id", "construct"], as_index=False)
        .agg(
            mean_answer=("answer_value", "mean"),
            std_answer=("answer_value", "std"),
            median_answer=("answer_value", "median"),
            n_responses=("answer_value", "count"),
            n_agents=("id_persona", "nunique"),
            mean_confidence=("confidence", "mean"),
        )
    )

    for col in ["mean_answer", "std_answer", "median_answer", "mean_confidence"]:
        out[col] = out[col].round(3)

    if output_path:
        out.to_csv(output_path, index=False, sep=";", encoding="utf-8-sig")

    return out


def aggregate_by_variables_construct(
    df_responses: pd.DataFrame,
    group_variables: list[str] | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    df = df_responses.copy()
    df["answer_value"] = pd.to_numeric(df["answer_value"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    group_variables = group_variables or DEFAULT_GROUP_VARIABLES
    tables = []

    for group_var in group_variables:
        if group_var not in df.columns:
            continue

        temp = (
            df
            .groupby(["artifact_id", "construct", group_var], as_index=False)
            .agg(
                mean_answer=("answer_value", "mean"),
                std_answer=("answer_value", "std"),
                median_answer=("answer_value", "median"),
                n_responses=("answer_value", "count"),
                n_agents=("id_persona", "nunique"),
                mean_confidence=("confidence", "mean"),
            )
            .rename(columns={group_var: "group_value"})
        )

        temp["group_variable"] = group_var
        tables.append(temp)

    out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    if not out.empty:
        for col in ["mean_answer", "std_answer", "median_answer", "mean_confidence"]:
            out[col] = out[col].round(3)

        out = out[
            [
                "artifact_id",
                "construct",
                "group_variable",
                "group_value",
                "mean_answer",
                "std_answer",
                "median_answer",
                "n_responses",
                "n_agents",
                "mean_confidence",
            ]
        ]

    if output_path:
        out.to_csv(output_path, index=False, sep=";", encoding="utf-8-sig")

    return out


def aggregate_by_agent_construct(
    df_responses: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    df = df_responses.copy()
    df["answer_value"] = pd.to_numeric(df["answer_value"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    group_cols = [
        "id_persona",
        "artifact_id",
        "construct",
        "segmento_e_bolhas",
        "geração",
        "ideologia",
        "identificação política",
        "raça/cor",
        "gênero",
        "renda familiar mensal",
        "uf",
        "tipo_de_território",
        "fonte_familiar",
    ]

    existing_group_cols = [c for c in group_cols if c in df.columns]

    out = (
        df
        .groupby(existing_group_cols, as_index=False)
        .agg(
            mean_answer=("answer_value", "mean"),
            std_answer=("answer_value", "std"),
            n_repeats=("answer_value", "count"),
            mean_confidence=("confidence", "mean"),
        )
    )

    for col in ["mean_answer", "std_answer", "mean_confidence"]:
        out[col] = out[col].round(3)

    if output_path:
        out.to_csv(output_path, index=False, sep=";", encoding="utf-8-sig")

    return out


def build_pivot_segment_construct(
    df_variables_construct: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    out = (
        df_variables_construct[
            df_variables_construct["group_variable"] == "segmento_e_bolhas"
        ]
        .pivot_table(
            index="group_value",
            columns="construct",
            values="mean_answer",
            aggfunc="mean",
        )
        .round(2)
    )

    if output_path:
        out.to_csv(output_path, sep=";", encoding="utf-8-sig")

    return out
