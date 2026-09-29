
def build_stimulus_response_questionnaire(
    include_optional_valence_item=False,
    include_extended_credibility_item=False,
    include_classic_attractiveness_item=False,
    panas_version="short_validated",
):
    """
    Questionario de reacao ao estimulo/conteudo.

    Cada item e uma pergunta individual.
    Todos os itens retornam resposta numerica de 1 a 7.
    O score do constructo deve ser calculado depois como media dos itens.

    Constructos:
    1. Valencia
    2. Arousal
    3. Emocao positiva
    4. Emocao negativa
    5. Credibilidade da mensagem
    6. Atratividade da fonte
    """

    questionnaire = []

    valence_items = [
        ("VAL_01", "feliz", "infeliz"),
        ("VAL_02", "agradado(a)", "desagradado(a)"),
        ("VAL_03", "satisfeito(a)", "insatisfeito(a)"),
        ("VAL_04", "contente", "melancolico(a)"),
    ]

    if include_optional_valence_item:
        valence_items.append(("VAL_05", "esperancoso(a)", "desesperancoso(a)"))

    for question_id, positive_anchor, negative_anchor in valence_items:
        questionnaire.append({
            "question_id": question_id,
            "construct": "Valencia",
            "subconstruct": "pleasure_pad",
            "item": f"{positive_anchor}-{negative_anchor}",
            "question": "Apos ler o conteudo acima, minha reacao foi:",
            "response_instruction": (
                "Responda em uma escala bipolar de 1 a 7. "
                f"1 significa '{negative_anchor}' e 7 significa '{positive_anchor}'. "
                "Use valores intermediarios para indicar intensidade moderada."
            ),
            "scale_type": "semantic_differential_bipolar_7",
            "scale_min": 1,
            "scale_max": 7,
            "low_anchor": negative_anchor,
            "high_anchor": positive_anchor,
            "scale_labels": {
                1: negative_anchor,
                2: f"bastante {negative_anchor}",
                3: f"um pouco {negative_anchor}",
                4: "neutro / intermediario",
                5: f"um pouco {positive_anchor}",
                6: f"bastante {positive_anchor}",
                7: positive_anchor,
            },
            "aggregation": "mean",
            "method_note": (
                "Valencia medida por escala bipolar de diferencial semantico "
                "adaptada da dimensao pleasure do PAD de Mehrabian e Russell, "
                "em linha com seu uso em marketing por Donovan e Rossiter."
            ),
        })

    arousal_items = [
        ("ARO_01", "estimulado(a)", "relaxado(a)"),
        ("ARO_02", "excitado(a)", "calmo(a)"),
        ("ARO_03", "desperto(a)", "sonolento(a)"),
    ]

    for question_id, high_anchor, low_anchor in arousal_items:
        questionnaire.append({
            "question_id": question_id,
            "construct": "Arousal",
            "subconstruct": "arousal_pad",
            "item": f"{high_anchor}-{low_anchor}",
            "question": "Apos ler o conteudo acima, senti-me:",
            "response_instruction": (
                "Responda em uma escala bipolar de 1 a 7. "
                f"1 significa '{low_anchor}' e 7 significa '{high_anchor}'. "
                "Use valores intermediarios para indicar intensidade moderada."
            ),
            "scale_type": "semantic_differential_bipolar_7",
            "scale_min": 1,
            "scale_max": 7,
            "low_anchor": low_anchor,
            "high_anchor": high_anchor,
            "scale_labels": {
                1: low_anchor,
                2: f"bastante {low_anchor}",
                3: f"um pouco {low_anchor}",
                4: "neutro / intermediario",
                5: f"um pouco {high_anchor}",
                6: f"bastante {high_anchor}",
                7: high_anchor,
            },
            "aggregation": "mean",
            "method_note": "Arousal medido por subconjunto abreviado da dimensao arousal do PAD.",
        })

    if panas_version == "short_validated":
        positive_emotions = [
            ("POS_01", "alerta"),
            ("POS_02", "inspirado(a)"),
            ("POS_03", "determinado(a)"),
            ("POS_04", "atento(a)"),
            ("POS_05", "ativo(a)"),
        ]
        negative_emotions = [
            ("NEG_01", "chateado(a)"),
            ("NEG_02", "hostil"),
            ("NEG_03", "envergonhado(a)"),
            ("NEG_04", "nervoso(a)"),
            ("NEG_05", "com medo"),
        ]
    elif panas_version == "lean":
        positive_emotions = [
            ("POS_01", "inspirado(a)"),
            ("POS_02", "atento(a)"),
            ("POS_03", "ativo(a)"),
        ]
        negative_emotions = [
            ("NEG_01", "nervoso(a)"),
            ("NEG_02", "assustado(a)"),
        ]
    else:
        raise ValueError("panas_version deve ser 'short_validated' ou 'lean'.")

    for question_id, emotion in positive_emotions:
        questionnaire.append({
            "question_id": question_id,
            "construct": "Emocao positiva",
            "subconstruct": "positive_affect",
            "item": emotion,
            "question": f"Ao ler o conteudo acima, senti-me {emotion}.",
            "response_instruction": (
                "Responda em uma escala de 1 a 7. "
                "1 significa 'nada' e 7 significa 'extremamente'."
            ),
            "scale_type": "likert_unipolar_7",
            "scale_min": 1,
            "scale_max": 7,
            "low_anchor": "nada",
            "high_anchor": "extremamente",
            "scale_labels": {
                1: "nada",
                2: "muito pouco",
                3: "um pouco",
                4: "moderadamente",
                5: "bastante",
                6: "muito",
                7: "extremamente",
            },
            "aggregation": "mean",
            "method_note": (
                "Emocao positiva medida por adaptacao do PANAS/I-PANAS-SF "
                "em instrucao de estado apos exposicao ao conteudo."
            ),
        })

    for question_id, emotion in negative_emotions:
        questionnaire.append({
            "question_id": question_id,
            "construct": "Emocao negativa",
            "subconstruct": "negative_affect",
            "item": emotion,
            "question": f"Ao ler o conteudo acima, senti-me {emotion}.",
            "response_instruction": (
                "Responda em uma escala de 1 a 7. "
                "1 significa 'nada' e 7 significa 'extremamente'."
            ),
            "scale_type": "likert_unipolar_7",
            "scale_min": 1,
            "scale_max": 7,
            "low_anchor": "nada",
            "high_anchor": "extremamente",
            "scale_labels": {
                1: "nada",
                2: "muito pouco",
                3: "um pouco",
                4: "moderadamente",
                5: "bastante",
                6: "muito",
                7: "extremamente",
            },
            "aggregation": "mean",
            "method_note": (
                "Emocao negativa medida por adaptacao do PANAS/I-PANAS-SF "
                "em instrucao de estado apos exposicao ao conteudo."
            ),
        })

    credibility_items = [
        ("CRE_01", "preciso"),
        ("CRE_02", "autentico"),
        ("CRE_03", "crivel"),
    ]

    if include_extended_credibility_item:
        credibility_items.append(("CRE_04", "confiavel"))

    for question_id, adjective in credibility_items:
        questionnaire.append({
            "question_id": question_id,
            "construct": "Credibilidade da mensagem",
            "subconstruct": "message_credibility",
            "item": adjective,
            "question": f"O quanto o adjetivo '{adjective}' descreve o conteudo acima?",
            "response_instruction": (
                "Responda em uma escala de 1 a 7. "
                "1 significa 'descreve muito mal' e 7 significa 'descreve muito bem'."
            ),
            "scale_type": "descriptive_likert_7",
            "scale_min": 1,
            "scale_max": 7,
            "low_anchor": "descreve muito mal",
            "high_anchor": "descreve muito bem",
            "scale_labels": {
                1: "descreve muito mal",
                2: "descreve mal",
                3: "descreve um pouco mal",
                4: "neutro / intermediario",
                5: "descreve um pouco bem",
                6: "descreve bem",
                7: "descreve muito bem",
            },
            "aggregation": "mean",
            "method_note": (
                "Credibilidade da mensagem medida pela Message Credibility Scale "
                "de Appelman e Sundar; forma principal com 3 itens."
            ),
        })

    attractiveness_items = [
        ("ATT_01", "atraente", "pouco atraente"),
        ("ATT_02", "elegante", "sem elegancia"),
        ("ATT_03", "bonito(a)", "feio(a)"),
        ("ATT_04", "tem classe", "nao tem classe"),
    ]

    if include_classic_attractiveness_item:
        attractiveness_items.append(("ATT_05", "sexy", "nao sexy"))

    for question_id, high_anchor, low_anchor in attractiveness_items:
        questionnaire.append({
            "question_id": question_id,
            "construct": "Atratividade da fonte",
            "subconstruct": "source_attractiveness",
            "item": f"{high_anchor}-{low_anchor}",
            "question": "A pessoa que aparece ou assina o conteudo acima me parece:",
            "response_instruction": (
                "Responda em uma escala bipolar de 1 a 7. "
                f"1 significa '{low_anchor}' e 7 significa '{high_anchor}'. "
                "Caso nao haja pessoa identificavel, responda pela fonte, porta-voz ou assinatura do conteudo."
            ),
            "scale_type": "semantic_differential_bipolar_7",
            "scale_min": 1,
            "scale_max": 7,
            "low_anchor": low_anchor,
            "high_anchor": high_anchor,
            "scale_labels": {
                1: low_anchor,
                2: f"bastante {low_anchor}",
                3: f"um pouco {low_anchor}",
                4: "neutro / intermediario",
                5: f"um pouco {high_anchor}",
                6: f"bastante {high_anchor}",
                7: high_anchor,
            },
            "aggregation": "mean",
            "method_note": (
                "Atratividade da fonte medida por versao abreviada da subescala "
                "attractiveness de Ohanian, adequada ao contexto do estimulo."
            ),
        })

    return questionnaire


def aggregate_construct_scores(df_responses):
    import pandas as pd

    df = df_responses.copy()
    df["answer_value"] = pd.to_numeric(df["answer_value"], errors="coerce")

    out = (
        df
        .groupby(["artifact_id", "id_persona", "construct"], as_index=False)
        .agg(
            construct_score=("answer_value", "mean"),
            construct_std=("answer_value", "std"),
            n_items=("answer_value", "count"),
        )
    )

    out["construct_score"] = out["construct_score"].round(3)
    out["construct_std"] = out["construct_std"].round(3)

    return out


def aggregate_construct_scores_by_group(df_responses, group_cols=None):
    import pandas as pd

    if group_cols is None:
        group_cols = ["segmento_e_bolhas"]

    df = df_responses.copy()
    df["answer_value"] = pd.to_numeric(df["answer_value"], errors="coerce")

    group_cols = [c for c in group_cols if c in df.columns]

    out = (
        df
        .groupby(["artifact_id", "construct", *group_cols], as_index=False)
        .agg(
            construct_score=("answer_value", "mean"),
            construct_std=("answer_value", "std"),
            n_responses=("answer_value", "count"),
            n_agents=("id_persona", "nunique"),
        )
    )

    out["construct_score"] = out["construct_score"].round(3)
    out["construct_std"] = out["construct_std"].round(3)

    return out


def pivot_construct_scores(df_construct_scores, index="id_persona"):
    return (
        df_construct_scores
        .pivot_table(
            index=index,
            columns="construct",
            values="construct_score",
            aggfunc="mean",
        )
        .reset_index()
    )
