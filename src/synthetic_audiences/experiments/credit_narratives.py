from __future__ import annotations

from typing import Any

import pandas as pd

from synthetic_audiences.settings import (
    SOCIO_DEMOGRAPHIC_COLUMNS,
    VALUES_ATTITUDES_PERCEPTIONS_COLUMNS,
)
from synthetic_audiences.utils import deterministic_id, stable_float_0_1


FINANCIAL_CONTEXT_COLUMNS = [
    "perfil_credito",
    "perfil_credito_label",
    "perfil_credito_subtipo",
    "perfil_profissional_experimental",
    "situacao_endividamento",
    "comprometimento_renda",
    "relacao_com_credito",
    "produto_credito_principal",
    "relacao_bancaria",
    "banco_digital_principal",
    "experiencia_inadimplencia",
    "literacia_financeira",
    "vulnerabilidade_financeira",
]


def safe_value(row: pd.Series, column: str, default: str = "Não informado") -> str:
    if column not in row.index:
        return default
    value = row[column]
    if pd.isna(value) or str(value).strip() == "":
        return default
    return str(value)


def _mask_clause(df: pd.DataFrame, clause: dict[str, Any]) -> pd.Series:
    column = clause["column"]
    if column not in df.columns:
        return pd.Series(False, index=df.index)

    series = df[column].fillna("").astype(str)
    if "in" in clause:
        return series.isin([str(v) for v in clause["in"]])
    if "not_in" in clause:
        return ~series.isin([str(v) for v in clause["not_in"]])
    if "equals" in clause:
        return series.eq(str(clause["equals"]))
    if "contains_any" in clause:
        terms = [str(v).lower() for v in clause["contains_any"]]
        lowered = series.str.lower()
        mask = pd.Series(False, index=df.index)
        for term in terms:
            mask = mask | lowered.str.contains(term, regex=False)
        return mask
    raise ValueError(f"Cláusula de elegibilidade não suportada: {clause}")


def eligible_agents(df: pd.DataFrame, profile: dict[str, Any]) -> pd.DataFrame:
    eligibility = profile.get("eligibility", {})
    mask = pd.Series(True, index=df.index)

    for clause in eligibility.get("all", []):
        mask = mask & _mask_clause(df, clause)

    any_clauses = eligibility.get("any", [])
    if any_clauses:
        any_mask = pd.Series(False, index=df.index)
        for clause in any_clauses:
            any_mask = any_mask | _mask_clause(df, clause)
        mask = mask & any_mask

    return df.loc[mask].copy()


def add_credit_profile_overlays(
    df_agents: pd.DataFrame,
    profiles_config: dict[str, Any],
    *,
    n_per_profile: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Select base personas and add the seven Itaú credit-profile overlays."""
    if n_per_profile <= 0:
        raise ValueError("n_per_profile deve ser maior que zero.")

    rows: list[dict[str, Any]] = []
    profile_defs = profiles_config["profiles"]

    for profile_index, profile in enumerate(profile_defs):
        profile_id = profile["id"]
        candidates = eligible_agents(df_agents, profile)
        fallback_used = False

        if len(candidates) < n_per_profile:
            candidates = df_agents.copy()
            fallback_used = True

        replace = len(candidates) < n_per_profile
        selected = candidates.sample(
            n=n_per_profile,
            replace=replace,
            random_state=seed + profile_index,
        ).reset_index(drop=True)

        subtypes = profile.get("subtypes") or [None]

        for i, (_, row) in enumerate(selected.iterrows()):
            record = row.to_dict()
            base_id = str(record.get("id_persona", f"base_{i}"))
            record["id_persona_base"] = base_id
            record["id_persona"] = deterministic_id(
                "credit_agent",
                {
                    "base_id": base_id,
                    "profile": profile_id,
                    "i": i,
                    "seed": seed,
                },
                digits=14,
            )
            record["perfil_credito"] = profile_id
            record["perfil_credito_label"] = profile["label"]
            record["profile_selection_fallback"] = fallback_used

            overlay = dict(profile.get("overlay", {}))
            subtype = subtypes[i % len(subtypes)] if subtypes else None
            if subtype:
                record["perfil_credito_subtipo"] = subtype["id"]
                overlay.update(subtype.get("overlay", {}))
            else:
                record["perfil_credito_subtipo"] = ""

            record.update(overlay)
            rows.append(record)

    return pd.DataFrame(rows)


def build_agent_financial_context(row: pd.Series) -> str:
    lines = []
    for column in FINANCIAL_CONTEXT_COLUMNS:
        value = safe_value(row, column, "")
        if value:
            lines.append(f"- {column}: {value}")
    return "\n".join(lines)


def build_base_agent_context(row: pd.Series) -> str:
    persona = safe_value(row, "Persona")
    socio = "\n".join(
        f"- {col}: {safe_value(row, col)}"
        for col in SOCIO_DEMOGRAPHIC_COLUMNS
        if col in row.index
    )
    values = "\n".join(
        f"- {col}: {safe_value(row, col)}"
        for col in VALUES_ATTITUDES_PERCEPTIONS_COLUMNS
        if col in row.index
    )

    return f"""PERSONA
{persona}

VARIÁVEIS SOCIO-DEMOGRÁFICAS
{socio}

VALORES, ATITUDES E PERCEPÇÕES
{values}

CONTEXTO FINANCEIRO E RELAÇÃO COM CRÉDITO
{build_agent_financial_context(row)}
""".strip()


def build_response_schema(
    items: list[dict[str, Any]],
    qualitative: dict[str, Any],
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []

    for item in items:
        item_id = item["id"]
        properties[item_id] = {
            "type": "integer",
            "minimum": int(item.get("scale_min", 1)),
            "maximum": int(item.get("scale_max", 7)),
        }
        required.append(item_id)

    categories = qualitative["categories"]
    properties["interpretation_category"] = {
        "type": "string",
        "enum": categories,
    }
    properties["interpretation_open"] = {"type": "string", "minLength": 1}
    properties["critique_open"] = {"type": "string", "minLength": 1}
    properties["overall_rationale"] = {"type": "string", "minLength": 1}
    properties["confidence"] = {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
    }
    required.extend(
        [
            "interpretation_category",
            "interpretation_open",
            "critique_open",
            "overall_rationale",
            "confidence",
        ]
    )

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_experiment_messages(
    agent_row: pd.Series,
    stimulus: dict[str, Any],
    items: list[dict[str, Any]],
    qualitative: dict[str, Any],
) -> list[dict[str, str]]:
    item_text = []
    for item in items:
        item_text.append(
            f"{item['id']} | {item['construct']}\n"
            f"Pergunta: {item['question']}\n"
            f"Escala: 1 = {item.get('low_anchor', 'discordo totalmente')} | "
            f"7 = {item.get('high_anchor', 'concordo totalmente')}"
        )

    categories = ", ".join(qualitative["categories"])
    human = f"""CONTEXTO DO RESPONDENTE
{build_base_agent_context(agent_row)}

ESTÍMULO
Tema: {stimulus['theme']}
Condição experimental: {stimulus['condition']}
Título: {stimulus['title']}

Conteúdo:
{stimulus['content']}

QUESTIONÁRIO
Responda a todos os itens abaixo como essa pessoa responderia imediatamente após a exposição ao estímulo.
Use apenas números inteiros de 1 a 7 nos itens quantitativos.

{chr(10).join(item_text)}

INTERPRETAÇÃO QUALITATIVA
{qualitative['interpretation_prompt']}
Escolha exatamente uma categoria entre: {categories}

Resposta aberta:
{qualitative['open_prompt']}

Crítica principal:
{qualitative['critique_prompt']}

Retorne apenas o JSON definido pelo schema.
""".strip()

    system = """Você é um simulador de respondentes sintéticos para pesquisa de comunicação e reputação.
Responda como uma pessoa real com o perfil fornecido, não como especialista, pesquisador, consultor ou modelo de IA.

Regras:
1. Use a Persona como guia principal e as demais variáveis como condicionantes.
2. O bloco financeiro descreve uma situação experimental adicional e deve ser levado a sério.
3. Não presuma que a mensagem da empresa é verdadeira apenas porque foi apresentada.
4. Não tente agradar a marca nem o pesquisador.
5. Diferencie confiança, justiça, transparência, risco e interesse comercial.
6. Não exponha raciocínio passo a passo.
7. A justificativa geral deve ter no máximo duas frases.
8. Retorne somente JSON válido.
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": human},
    ]


def mock_experiment_response(
    agent_id: str,
    stimulus: dict[str, Any],
    items: list[dict[str, Any]],
    qualitative: dict[str, Any],
    iteration: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        u = stable_float_0_1(
            agent_id,
            stimulus["id"],
            item["id"],
            iteration,
        )
        result[item["id"]] = max(1, min(7, int(u * 7) + 1))

    ucat = stable_float_0_1(agent_id, stimulus["id"], "category", iteration)
    categories = qualitative["categories"]
    idx = min(int(ucat * len(categories)), len(categories) - 1)
    category = categories[idx]

    result["interpretation_category"] = category
    result["interpretation_open"] = (
        f"Resposta mock: interpretação classificada como {category}."
    )
    result["critique_open"] = (
        "Resposta mock para validar persistência, agregação e leitura qualitativa."
    )
    result["overall_rationale"] = "Resposta mock determinística; não possui validade analítica."
    result["confidence"] = round(
        0.55 + stable_float_0_1(agent_id, stimulus["id"], iteration) * 0.3,
        3,
    )
    return result


def response_to_long_rows(
    agent_row: pd.Series,
    stimulus: dict[str, Any],
    items: list[dict[str, Any]],
    response: dict[str, Any],
    *,
    run_id: str,
    model: str,
    iteration: int,
    is_mock: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    metadata_columns = [
        "id_persona",
        "id_persona_base",
        "perfil_credito",
        "perfil_credito_label",
        "perfil_credito_subtipo",
        "segmento_e_bolhas",
        "geração",
        "ideologia",
        "identificação política",
        "gênero",
        "idade",
        "renda familiar mensal",
        "situação ocupacional",
        "uf",
        "tipo_de_território",
        "literacia_financeira",
        "vulnerabilidade_financeira",
    ]
    metadata = {
        col: safe_value(agent_row, col, "")
        for col in metadata_columns
        if col in agent_row.index
    }

    for item in items:
        score = int(response[item["id"]])
        direction = item.get("direction", "positive")
        favorable = score if direction == "positive" else 8 - score

        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "is_mock": is_mock,
                "theme": stimulus["theme"],
                "stimulus_id": stimulus["id"],
                "condition": stimulus["condition"],
                "repeat_id": iteration,
                **metadata,
                "item_id": item["id"],
                "construct": item["construct"],
                "direction": direction,
                "score_raw": score,
                "score_favorable": favorable,
                "confidence": float(response.get("confidence", 0.5)),
                "interpretation_category": response.get("interpretation_category", ""),
                "interpretation_open": response.get("interpretation_open", ""),
                "critique_open": response.get("critique_open", ""),
                "overall_rationale": response.get("overall_rationale", ""),
            }
        )

    return rows
