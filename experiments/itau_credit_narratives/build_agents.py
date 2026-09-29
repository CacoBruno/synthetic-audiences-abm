from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_audiences.experiments.credit_narratives import add_credit_profile_overlays


HERE = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) == 1:
        retry = pd.read_csv(path, sep=";")
        if len(retry.columns) > 1:
            return retry
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Constrói os 7 perfis financeiros para o experimento Itaú."
    )
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--n-per-profile", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml(HERE / "config.yml")
    profiles = load_yaml(HERE / "profiles.yml")
    exp = config["experiment"]

    source = args.source or (ROOT / config["paths"]["agents_source"])
    n_per_profile = args.n_per_profile or int(exp["n_per_profile"])
    seed = args.seed if args.seed is not None else int(exp["seed"])
    output = args.output or (HERE / "outputs" / "agents_itau_credit.csv")

    if not source.exists():
        raise FileNotFoundError(
            f"Base de personas não encontrada: {source}. "
            "Confirme que cluster_perfil_persona.csv está na raiz do projeto."
        )

    base = read_csv_auto(source)
    agents = add_credit_profile_overlays(
        base,
        profiles,
        n_per_profile=n_per_profile,
        seed=seed,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    agents.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"Base de origem: {source}")
    print(f"Agentes experimentais: {len(agents)}")
    print(f"Perfis: {agents['perfil_credito'].nunique()}")
    print(agents["perfil_credito"].value_counts().sort_index().to_string())
    print(f"Fallback de seleção: {int(agents['profile_selection_fallback'].sum())} agentes")
    print(f"Arquivo salvo em: {output}")


if __name__ == "__main__":
    main()
