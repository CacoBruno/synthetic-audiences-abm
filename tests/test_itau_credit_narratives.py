import pandas as pd

from synthetic_audiences.experiments.credit_narratives import (
    add_credit_profile_overlays,
    build_response_schema,
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
    }

    schema = build_response_schema(items, qualitative)

    assert schema["properties"]["Q1"]["minimum"] == 1
    assert schema["properties"]["Q1"]["maximum"] == 7
    assert set(["Q1", "Q2"]).issubset(schema["required"])
    assert schema["properties"]["interpretation_category"]["enum"] == ["a", "b"]


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
        "interpretation_category": "x",
        "interpretation_open": "x",
        "critique_open": "x",
        "overall_rationale": "x",
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
