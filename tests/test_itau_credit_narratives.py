import pandas as pd
import pytest

from synthetic_audiences.experiments.credit_narratives import (
    add_credit_profile_overlays,
    balanced_limit_agents,
    build_response_schema,
    questionnaire_for_condition,
    response_to_long_rows,
)


def test_credit_profile_overlay_preserves_base_agent():
    base = pd.DataFrame(
        [
            {
                "id_persona": "agent_1",
                "renda familiar mensal": "Até 1 salário mínimo",
                "idade": "35-49 anos",
                "Persona": "Pessoa sintética de teste.",
            }
        ]
    )
    config = {
        "profiles": [
            {
                "id": "endividado_baixa_renda",
                "label": "Endividado de baixa renda",
                "eligibility": {
                    "all": [
                        {
                            "column": "renda familiar mensal",
                            "in": ["Até 1 salário mínimo"],
                        }
                    ]
                },
                "overlay": {
                    "situacao_endividamento": "Dívidas ativas.",
                    "vulnerabilidade_financeira": "Alta.",
                },
            }
        ]
    }

    out = add_credit_profile_overlays(base, config, n_per_profile=1, seed=42)

    assert len(out) == 1
    assert out.loc[0, "id_persona_base"] == "agent_1"
    assert out.loc[0, "id_persona"] != "agent_1"
    assert out.loc[0, "perfil_credito"] == "endividado_baixa_renda"
    assert out.loc[0, "vulnerabilidade_financeira"] == "Alta."


def test_response_schema_requires_every_item_and_qualitative_fields():
    items = [
        {
            "id": "Q1",
            "construct": "Teste",
            "scale_min": 1,
            "scale_max": 7,
        },
        {
            "id": "Q2",
            "construct": "Teste 2",
            "scale_min": 1,
            "scale_max": 7,
        },
    ]
    qualitative = {
        "categories": ["a", "b"],
        "residual_concern_categories": ["none", "risk"],
    }

    schema = build_response_schema(items, qualitative)

    assert schema["properties"]["Q1"]["minimum"] == 1
    assert schema["properties"]["Q1"]["maximum"] == 7
    assert set(["Q1", "Q2"]).issubset(schema["required"])
    assert schema["properties"]["primary_interpretation"]["enum"] == ["a", "b"]
    assert schema["properties"]["residual_concern"]["enum"] == ["none", "risk"]
    assert "motivation_summary" in schema["required"]


def test_negative_items_are_reversed_only_in_favorable_score():
    agent = pd.Series(
        {
            "id_persona": "agent_test",
            "perfil_credito": "teste",
            "perfil_credito_label": "Teste",
        }
    )
    stimulus = {
        "id": "stim_1",
        "theme": "tema",
        "condition": "narrative",
    }
    items = [
        {
            "id": "POS",
            "construct": "Positivo",
            "direction": "positive",
        },
        {
            "id": "NEG",
            "construct": "Risco",
            "direction": "negative",
        },
    ]
    response = {
        "POS": 6,
        "NEG": 6,
        "confidence": 0.8,
        "primary_interpretation": "x",
        "residual_concern": "none",
        "interpretation_open": "x",
        "critique_open": "x",
        "motivation_summary": "porque x",
    }

    rows = response_to_long_rows(
        agent,
        stimulus,
        items,
        response,
        run_id="run",
        model="mock",
        iteration=0,
        is_mock=True,
    )
    by_item = {row["item_id"]: row for row in rows}

    assert by_item["POS"]["score_raw"] == 6
    assert by_item["POS"]["score_favorable"] == 6
    assert by_item["NEG"]["score_raw"] == 6
    assert by_item["NEG"]["score_favorable"] == 2


def test_credit_profile_overlay_refuses_generic_fallback():
    base = pd.DataFrame(
        [
            {
                "id_persona": "agent_1",
                "renda familiar mensal": "Mais de 20 salários mínimos",
                "Persona": "Pessoa sintética de teste.",
            }
        ]
    )
    config = {
        "profiles": [
            {
                "id": "endividado_baixa_renda",
                "label": "Endividado de baixa renda",
                "eligibility": {
                    "all": [
                        {
                            "column": "renda familiar mensal",
                            "in": ["Até 1 salário mínimo"],
                        }
                    ]
                },
                "overlay": {},
            }
        ]
    }

    with pytest.raises(ValueError, match="endividado_baixa_renda"):
        add_credit_profile_overlays(base, config, n_per_profile=1, seed=42)


def test_balanced_limit_agents_selects_one_per_profile():
    rows = []
    for profile in ["a", "b", "c", "d", "e", "f", "g"]:
        for i in range(2):
            rows.append(
                {
                    "id_persona": f"{profile}_{i}",
                    "perfil_credito": profile,
                }
            )
    df = pd.DataFrame(rows)

    sampled = balanced_limit_agents(df, 7, seed=42)

    counts = sampled["perfil_credito"].value_counts().to_dict()
    assert len(sampled) == 7
    assert counts == {profile: 1 for profile in ["a", "b", "c", "d", "e", "f", "g"]}


def test_questionnaire_is_condition_aware():
    questionnaire = {
        "common_items": [
            {"id": "ALL", "construct": "all"},
            {
                "id": "BRAND",
                "construct": "brand",
                "conditions": ["narrative", "context_plus_narrative"],
            },
        ],
        "themes": {
            "tema": {
                "items": [
                    {
                        "id": "CTX",
                        "construct": "context",
                        "conditions": ["context"],
                    },
                    {
                        "id": "MSG",
                        "construct": "message",
                        "conditions": ["narrative", "context_plus_narrative"],
                    },
                ],
                "qualitative_by_condition": {
                    "context": {
                        "categories": ["contexto"],
                        "interpretation_prompt": "contexto",
                        "open_prompt": "contexto",
                        "critique_prompt": "contexto",
                    },
                    "narrative": {
                        "categories": ["mensagem"],
                        "interpretation_prompt": "mensagem",
                        "open_prompt": "mensagem",
                        "critique_prompt": "mensagem",
                    },
                },
            }
        },
    }

    context_items, context_qual = questionnaire_for_condition(
        questionnaire, "tema", "context"
    )
    narrative_items, narrative_qual = questionnaire_for_condition(
        questionnaire, "tema", "narrative"
    )

    assert [item["id"] for item in context_items] == ["ALL", "CTX"]
    assert [item["id"] for item in narrative_items] == ["ALL", "BRAND", "MSG"]
    assert context_qual["categories"] == ["contexto"]
    assert narrative_qual["categories"] == ["mensagem"]


def test_long_rows_persist_qualitative_explanation_fields():
    agent = pd.Series(
        {
            "id_persona": "agent_test",
            "perfil_credito": "teste",
            "perfil_credito_label": "Teste",
        }
    )
    stimulus = {"id": "stim_1", "theme": "tema", "condition": "narrative"}
    items = [{"id": "Q1", "construct": "Teste", "direction": "positive"}]
    response = {
        "Q1": 5,
        "confidence": 0.8,
        "primary_interpretation": "capacidade_pagamento",
        "residual_concern": "falta_transparencia",
        "interpretation_open": "Entendi a proposta.",
        "critique_open": "Ainda faltam detalhes.",
        "motivation_summary": "A proposta parece útil, mas quero saber como a decisão é tomada.",
    }

    rows = response_to_long_rows(
        agent,
        stimulus,
        items,
        response,
        run_id="run",
        model="mock",
        iteration=0,
        is_mock=True,
    )

    row = rows[0]
    assert row["primary_interpretation"] == "capacidade_pagamento"
    assert row["interpretation_category"] == "capacidade_pagamento"
    assert row["residual_concern"] == "falta_transparencia"
    assert "como a decisão" in row["motivation_summary"]
