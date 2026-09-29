from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"


def resolve_run_dir(requested: Path | None) -> Path:
    if requested:
        return requested.resolve()
    candidates = sorted(
        [p for p in OUTPUTS.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Nenhum run encontrado em outputs/.")
    return candidates[0]


def compute_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    call_keys = [
        "run_id",
        "id_persona",
        "theme",
        "stimulus_id",
        "condition",
        "repeat_id",
        "perfil_credito",
        "perfil_credito_label",
    ]

    per_call_construct = (
        df.groupby([*call_keys, "construct"], as_index=False)
        .agg(
            construct_score_raw=("score_raw", "mean"),
            construct_score_favorable=("score_favorable", "mean"),
            confidence=("confidence", "mean"),
        )
    )

    construct_summary = (
        per_call_construct.groupby(
            ["theme", "condition", "perfil_credito", "perfil_credito_label", "construct"],
            as_index=False,
        )
        .agg(
            mean_raw=("construct_score_raw", "mean"),
            mean_favorable=("construct_score_favorable", "mean"),
            std_favorable=("construct_score_favorable", "std"),
            n_agents=("id_persona", "nunique"),
            n_observations=("construct_score_favorable", "count"),
            mean_confidence=("confidence", "mean"),
        )
    )

    per_call_overall = (
        per_call_construct.groupby(call_keys, as_index=False)
        .agg(
            overall_favorable=("construct_score_favorable", "mean"),
            confidence=("confidence", "mean"),
        )
    )

    overall_summary = (
        per_call_overall.groupby(
            ["theme", "condition", "perfil_credito", "perfil_credito_label"],
            as_index=False,
        )
        .agg(
            overall_favorable=("overall_favorable", "mean"),
            overall_std=("overall_favorable", "std"),
            n_agents=("id_persona", "nunique"),
            n_observations=("overall_favorable", "count"),
            mean_confidence=("confidence", "mean"),
        )
    )

    # Primary message comparison: narrative alone vs. context + narrative.
    # Both conditions use the same brand-evaluation item set.
    message_resilience = overall_summary.pivot_table(
        index=["theme", "perfil_credito", "perfil_credito_label"],
        columns="condition",
        values="overall_favorable",
        aggfunc="mean",
    ).reset_index()

    for column in ["narrative", "context_plus_narrative"]:
        if column not in message_resilience.columns:
            message_resilience[column] = pd.NA

    message_resilience["resilience"] = (
        message_resilience["context_plus_narrative"]
        - message_resilience["narrative"]
    )

    # Construct-level comparison. context_delta is only populated when that
    # construct is valid in both the context and context+narrative conditions.
    construct_comparison = construct_summary.pivot_table(
        index=["theme", "perfil_credito", "perfil_credito_label", "construct"],
        columns="condition",
        values="mean_favorable",
        aggfunc="mean",
    ).reset_index()

    for column in ["context", "narrative", "context_plus_narrative"]:
        if column not in construct_comparison.columns:
            construct_comparison[column] = pd.NA

    construct_comparison["resilience"] = (
        construct_comparison["context_plus_narrative"]
        - construct_comparison["narrative"]
    )
    construct_comparison["context_delta"] = (
        construct_comparison["context_plus_narrative"]
        - construct_comparison["context"]
    )

    context_diagnostics = construct_summary[
        construct_summary["condition"] == "context"
    ].copy()

    qualitative = df.drop_duplicates(
        subset=[
            "run_id",
            "id_persona",
            "theme",
            "stimulus_id",
            "condition",
            "repeat_id",
        ]
    ).copy()

    interpretation_shares = (
        qualitative.groupby(
            [
                "theme",
                "condition",
                "perfil_credito",
                "perfil_credito_label",
                "interpretation_category",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n"})
    )
    totals = (
        interpretation_shares.groupby(
            ["theme", "condition", "perfil_credito"],
            as_index=False,
        )["n"]
        .sum()
        .rename(columns={"n": "total"})
    )
    interpretation_shares = interpretation_shares.merge(
        totals,
        on=["theme", "condition", "perfil_credito"],
        how="left",
    )
    interpretation_shares["share"] = (
        interpretation_shares["n"] / interpretation_shares["total"]
    )

    open_responses = qualitative[
        [
            "theme",
            "condition",
            "perfil_credito",
            "perfil_credito_label",
            "perfil_credito_subtipo",
            "id_persona",
            "interpretation_category",
            "interpretation_open",
            "critique_open",
            "overall_rationale",
            "confidence",
        ]
    ].copy()

    item_summary = (
        df.groupby(
            [
                "theme",
                "condition",
                "perfil_credito",
                "perfil_credito_label",
                "item_id",
                "construct",
                "direction",
            ],
            as_index=False,
        )
        .agg(
            mean_raw=("score_raw", "mean"),
            mean_favorable=("score_favorable", "mean"),
            std_raw=("score_raw", "std"),
            n_agents=("id_persona", "nunique"),
        )
    )

    return {
        "item_summary": item_summary,
        "construct_summary": construct_summary,
        "overall_summary": overall_summary,
        "message_resilience": message_resilience,
        "construct_comparison": construct_comparison,
        "context_diagnostics": context_diagnostics,
        "interpretation_shares": interpretation_shares,
        "open_responses": open_responses,
    }


def render_report(tables: dict[str, pd.DataFrame], is_mock: bool) -> str:
    resilience = tables["message_resilience"].copy()
    shares = tables["interpretation_shares"].copy()

    lines = ["# Itaú — Teste de narrativas com audiências sintéticas", ""]
    if is_mock:
        lines.extend(
            [
                "> **SMOKE TEST MOCK:** os números abaixo são determinísticos e servem apenas para validar o pipeline. Não possuem validade analítica.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Os resultados resumem a reação das audiências financeiras às condições experimentais.",
                "",
                "- **Context** é diagnóstico do problema e não deve ser interpretado como mensagem do Itaú.",
                "- **Resiliência** = contexto + narrativa − narrativa isolada.",
                "- Resiliência negativa indica perda de desempenho quando a controvérsia é conhecida; positiva indica manutenção ou ganho.",
                "- Scores favoráveis estão em escala 1–7; itens de risco são invertidos apenas na métrica favorável.",
                "",
            ]
        )

    for theme in resilience["theme"].dropna().unique():
        lines.append(f"## {theme}")
        lines.append("")

        theme_resilience = resilience[resilience["theme"] == theme].copy()
        cols = [
            "perfil_credito_label",
            "narrative",
            "context_plus_narrative",
            "resilience",
        ]
        rounded = theme_resilience[cols].copy()
        numeric_cols = [c for c in cols if c != "perfil_credito_label"]
        rounded[numeric_cols] = (
            rounded[numeric_cols]
            .apply(pd.to_numeric, errors="coerce")
            .round(2)
        )
        lines.append("### Desempenho da narrativa")
        lines.append("")
        lines.append(rounded.to_markdown(index=False))
        lines.append("")

        for condition, title in [
            ("context", "Leitura predominante do contexto"),
            ("context_plus_narrative", "Leitura predominante após contexto + narrativa"),
        ]:
            selected = shares[
                (shares["theme"] == theme)
                & (shares["condition"] == condition)
            ].copy()
            if selected.empty:
                continue
            top = (
                selected.sort_values(
                    ["perfil_credito_label", "share"],
                    ascending=[True, False],
                )
                .groupby("perfil_credito_label", as_index=False)
                .head(1)
            )
            top["share"] = (top["share"] * 100).round(1)
            lines.append(f"### {title}")
            lines.append("")
            lines.append(
                top[
                    [
                        "perfil_credito_label",
                        "interpretation_category",
                        "share",
                    ]
                ]
                .rename(columns={"share": "share_%"})
                .to_markdown(index=False)
            )
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agrega e analisa o experimento Itaú.")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    source = run_dir / "responses_long.csv"
    if not source.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {source}")

    df = pd.read_csv(source)
    tables = compute_tables(df)

    for name, table in tables.items():
        table.to_csv(
            run_dir / f"{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    is_mock = bool(df["is_mock"].astype(str).str.lower().eq("true").all())
    report = render_report(tables, is_mock=is_mock)
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    print(f"Run analisado: {run_dir}")
    print("Arquivos gerados:")
    for name in tables:
        print(f"- {name}.csv")
    print("- report.md")


if __name__ == "__main__":
    main()
