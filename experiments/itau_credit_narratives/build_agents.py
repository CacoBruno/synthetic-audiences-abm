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

from synthetic_audiences.experiments.credit_narratives import (
    add_credit_profile_overlays,
    eligible_agents,
)
from synthetic_audiences.population import generate_profiles


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
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "CSV opcional de personas individuais. Se omitido, gera uma "
            "população-base reproduzível com o gerador do projeto."
        ),
    )
    parser.add_argument("--base-population-size", type=int, default=None)
    parser.add_argument("--n-per-profile", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml(HERE / "config.yml")
    profiles = load_yaml(HERE / "profiles.yml")
    exp = config["experiment"]

    n_per_profile = args.n_per_profile or int(exp["n_per_profile"])
    base_population_size = (
        args.base_population_size or int(exp["base_population_size"])
    )
    seed = args.seed if args.seed is not None else int(exp["seed"])
    output = args.output or (HERE / "outputs" / "agents_itau_credit.csv")

    if args.source is not None:
        source = args.source.resolve()
        if not source.exists():
            raise FileNotFoundError(f"Base de personas não encontrada: {source}")
        base = read_csv_auto(source)
        source_label = str(source)
    else:
        population_config = ROOT / config["paths"]["population_config"]
        base = generate_profiles(
            n=base_population_size,
            config_path=population_config,
            seed=seed,
        )
        source_label = (
            f"população gerada pelo projeto "
            f"(n={base_population_size}, seed={seed}, config={population_config})"
        )

        generated_base_path = (
            HERE
            / "outputs"
            / f"base_population_n{base_population_size}_seed{seed}.csv"
        )
        generated_base_path.parent.mkdir(parents=True, exist_ok=True)
        base.to_csv(generated_base_path, index=False, encoding="utf-8-sig")

    print(f"Base de origem: {source_label}")
    print(f"População-base: {len(base)} agentes")
    print("Elegibilidade por perfil:")

    candidate_counts = {}
    for profile in profiles["profiles"]:
        n_candidates = len(eligible_agents(base, profile))
        candidate_counts[profile["id"]] = n_candidates
        print(f"- {profile['id']}: {n_candidates} candidatos")

    insufficient = {
        profile_id: count
        for profile_id, count in candidate_counts.items()
        if count < n_per_profile
    }
    if insufficient:
        details = ", ".join(
            f"{profile_id}={count}"
            for profile_id, count in insufficient.items()
        )
        raise ValueError(
            "População-base insuficiente para selecionar todos os perfis sem "
            f"fallback: {details}. Aumente --base-population-size ou revise "
            "profiles.yml."
        )

    agents = add_credit_profile_overlays(
        base,
        profiles,
        n_per_profile=n_per_profile,
        seed=seed,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    agents.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"Agentes experimentais: {len(agents)}")
    print(f"Perfis: {agents['perfil_credito'].nunique()}")
    print(agents["perfil_credito"].value_counts().sort_index().to_string())
    print(f"Fallback de seleção: {int(agents['profile_selection_fallback'].sum())}")
    print(f"Arquivo salvo em: {output}")


if __name__ == "__main__":
    main()
