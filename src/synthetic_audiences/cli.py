from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from synthetic_audiences.config import DEFAULT_CONFIG_PATH
from synthetic_audiences.io import read_artifact_json, read_questions_csv
from synthetic_audiences.population import generate_profiles
from synthetic_audiences.runner import apply_questionnaire, summarize_results

load_dotenv()
app = typer.Typer(help="Synthetic audiences ABM CLI")
console = Console()


@app.command("generate")
def generate_cmd(
    n: int = typer.Option(100, help="Número de agentes sintéticos."),
    output: Path = typer.Option(Path("data/output/agents.csv"), help="CSV de saída."),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, help="YAML de referência."),
    seed: int = typer.Option(42, help="Semente de reprodutibilidade."),
) -> None:
    """Generate synthetic audience profiles."""
    df = generate_profiles(n=n, output_csv=output, config_path=config, seed=seed)
    console.print(f"[green]OK[/green] {len(df)} agentes salvos em {output}")
    table = Table(title="Distribuição por segmento")
    table.add_column("Segmento")
    table.add_column("N")
    for segment, count in df["segmento_e_bolhas"].value_counts().items():
        table.add_row(segment, str(count))
    console.print(table)


@app.command("run")
def run_cmd(
    agents_csv: Path = typer.Option(Path("data/output/agents.csv"), help="CSV com agentes."),
    questions_csv: Path = typer.Option(Path("examples/questionnaire_example.csv"), help="CSV com questionário."),
    artifact_json: Path = typer.Option(Path("examples/artifact_example.json"), help="JSON com artefato."),
    output_jsonl: Path = typer.Option(Path("data/output/responses.jsonl"), help="JSONL de respostas."),
    summary_csv: Path = typer.Option(Path("data/output/summary.csv"), help="CSV agregado."),
    repeats: int = typer.Option(3, help="Número de aplicações por agente/pergunta."),
    mock: bool = typer.Option(False, help="Rodar sem OpenAI, com respostas determinísticas mock."),
    model: str | None = typer.Option(None, help="Modelo OpenAI. Usa OPENAI_MODEL se omitido."),
    limit_agents: int | None = typer.Option(None, help="Limitar agentes para teste."),
) -> None:
    """Apply a questionnaire to agents for one artifact."""
    agents = pd.read_csv(agents_csv)
    questions = read_questions_csv(questions_csv)
    artifact = read_artifact_json(artifact_json)
    if output_jsonl.exists():
        output_jsonl.unlink()
    results = apply_questionnaire(
        agents=agents,
        questions=questions,
        artifact=artifact,
        repeats=repeats,
        output_jsonl=output_jsonl,
        use_mock=mock,
        model=model,
        limit_agents=limit_agents,
    )
    summary = summarize_results(results, output_csv=summary_csv)
    console.print(f"[green]OK[/green] respostas salvas em {output_jsonl}")
    console.print(f"[green]OK[/green] resumo salvo em {summary_csv}")
    console.print(summary.head(10))


if __name__ == "__main__":
    app()
