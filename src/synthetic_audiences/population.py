from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd

from synthetic_audiences.config import DEFAULT_CONFIG_PATH, load_config
from synthetic_audiences.utils import deterministic_id, ensure_parent, weighted_choice


AGE_BY_GENERATION = {
    "Geração Bossa Nova": {"50-64 anos": 0.35, "65 anos ou mais": 0.65},
    "Geração Ordem e Progresso": {"35-49 anos": 0.35, "50-64 anos": 0.55, "65 anos ou mais": 0.10},
    "Geração Redemocratização": {"25-34 anos": 0.45, "35-49 anos": 0.45, "16-24 anos": 0.10},
    "Geração.Com": {"16-24 anos": 0.90, "25-34 anos": 0.10},
}

POLITICS_BY_SEGMENT = {
    "Militantes de esquerda": {"Esquerda": 0.70, "Centro-esquerda": 0.25, "Centro": 0.05},
    "Progressistas": {"Esquerda": 0.35, "Centro-esquerda": 0.45, "Centro": 0.20},
    "Dependentes do Estado / Classe D e E": {"Esquerda": 0.25, "Centro-esquerda": 0.25, "Centro": 0.35, "Centro-direita": 0.10, "Não se identifica politicamente": 0.05},
    "Liberais sociais": {"Centro": 0.40, "Centro-direita": 0.35, "Centro-esquerda": 0.15, "Direita": 0.10},
    "Empreendedores individuais": {"Centro": 0.30, "Centro-direita": 0.30, "Direita": 0.20, "Centro-esquerda": 0.10, "Não se identifica politicamente": 0.10},
    "Conservadores cristãos": {"Direita": 0.38, "Centro-direita": 0.35, "Centro": 0.20, "Não se identifica politicamente": 0.07},
    "Empresários": {"Direita": 0.45, "Centro-direita": 0.35, "Centro": 0.15, "Centro-esquerda": 0.05},
    "Agro": {"Direita": 0.50, "Centro-direita": 0.30, "Centro": 0.15, "Não se identifica politicamente": 0.05},
    "Extrema direita": {"Direita": 0.75, "Centro-direita": 0.20, "Centro": 0.05},
}

IDEOLOGY_BY_SEGMENT = {
    "Militantes de esquerda": {"Progressista": 0.55, "Muito progressista": 0.25, "Moderado(a)": 0.20},
    "Progressistas": {"Progressista": 0.60, "Muito progressista": 0.25, "Moderado(a)": 0.15},
    "Dependentes do Estado / Classe D e E": {"Moderado(a)": 0.40, "Conservador(a)": 0.35, "Progressista": 0.20, "Prefere não responder": 0.05},
    "Liberais sociais": {"Moderado(a)": 0.45, "Progressista": 0.25, "Conservador(a)": 0.20, "Muito progressista": 0.10},
    "Empreendedores individuais": {"Conservador(a)": 0.35, "Moderado(a)": 0.40, "Progressista": 0.15, "Muito conservador(a)": 0.10},
    "Conservadores cristãos": {"Conservador(a)": 0.55, "Muito conservador(a)": 0.25, "Moderado(a)": 0.20},
    "Empresários": {"Moderado(a)": 0.35, "Conservador(a)": 0.35, "Progressista": 0.20, "Muito conservador(a)": 0.10},
    "Agro": {"Conservador(a)": 0.55, "Muito conservador(a)": 0.20, "Moderado(a)": 0.25},
    "Extrema direita": {"Muito conservador(a)": 0.65, "Conservador(a)": 0.35},
}

EDUCATION_BY_SEGMENT = {
    "Progressistas": {"Ensino médio completo": 0.22, "Ensino superior incompleto": 0.22, "Ensino superior completo": 0.35, "Pós-graduação": 0.21},
    "Militantes de esquerda": {"Ensino médio completo": 0.20, "Ensino superior incompleto": 0.20, "Ensino superior completo": 0.35, "Pós-graduação": 0.25},
    "Empresários": {"Ensino médio completo": 0.15, "Ensino técnico / profissionalizante": 0.15, "Ensino superior completo": 0.45, "Pós-graduação": 0.25},
    "Liberais sociais": {"Ensino médio completo": 0.18, "Ensino superior incompleto": 0.22, "Ensino superior completo": 0.40, "Pós-graduação": 0.20},
    "Agro": {"Ensino fundamental completo": 0.15, "Ensino médio completo": 0.38, "Ensino técnico / profissionalizante": 0.20, "Ensino superior completo": 0.20, "Pós-graduação": 0.07},
    "Conservadores cristãos": {"Ensino fundamental completo": 0.18, "Ensino médio incompleto": 0.12, "Ensino médio completo": 0.42, "Ensino técnico / profissionalizante": 0.13, "Ensino superior completo": 0.12, "Pós-graduação": 0.03},
    "Dependentes do Estado / Classe D e E": {"Sem instrução": 0.04, "Ensino fundamental incompleto": 0.20, "Ensino fundamental completo": 0.22, "Ensino médio incompleto": 0.16, "Ensino médio completo": 0.30, "Ensino técnico / profissionalizante": 0.05, "Ensino superior incompleto": 0.03},
    "Empreendedores individuais": {"Ensino fundamental completo": 0.12, "Ensino médio incompleto": 0.12, "Ensino médio completo": 0.45, "Ensino técnico / profissionalizante": 0.16, "Ensino superior incompleto": 0.10, "Ensino superior completo": 0.05},
    "Extrema direita": {"Ensino médio completo": 0.35, "Ensino técnico / profissionalizante": 0.18, "Ensino superior incompleto": 0.12, "Ensino superior completo": 0.25, "Pós-graduação": 0.10},
}

OCCUPATION_BY_SEGMENT = {
    "Empresários": {"Empregador / empresário": 0.65, "Autônomo / conta própria": 0.10, "Empregado com carteira assinada": 0.15, "Funcionário público": 0.05, "Aposentado / pensionista": 0.05},
    "Empreendedores individuais": {"MEI": 0.38, "Autônomo / conta própria": 0.40, "Empregado sem carteira assinada": 0.12, "Empregado com carteira assinada": 0.10},
    "Agro": {"Autônomo / conta própria": 0.26, "Empregador / empresário": 0.20, "Empregado com carteira assinada": 0.28, "Empregado sem carteira assinada": 0.10, "MEI": 0.08, "Aposentado / pensionista": 0.08},
    "Dependentes do Estado / Classe D e E": {"Empregado sem carteira assinada": 0.18, "Autônomo / conta própria": 0.20, "Desempregado procurando trabalho": 0.18, "Dona(o) de casa / trabalho doméstico não remunerado": 0.18, "Empregado com carteira assinada": 0.16, "Aposentado / pensionista": 0.10},
}

SEGMENT_DESCRIPTIONS = {
    "Conservadores cristãos": "valoriza fé, ordem, família, respeito aos mais velhos e segurança pública, com forte presença de referências religiosas no cotidiano.",
    "Dependentes do Estado / Classe D e E": "vive com forte sensibilidade a preços, emprego, renda e políticas públicas, combinando necessidade material com valores familiares e comunitários.",
    "Agro": "tem vínculo com interior, produção rural, cultura sertaneja, empreendedorismo local e defesa de ordem, propriedade e baixa regulação.",
    "Progressistas": "valoriza diversidade, minorias, clima, igualdade e abertura de costumes, com maior proximidade de repertórios digitais e urbanos.",
    "Militantes de esquerda": "tem alto interesse político, leitura polarizada da disputa pública e defesa de direitos sociais, Estado e redistribuição.",
    "Empresários": "prioriza estabilidade econômica, liberdade de mercado, eficiência, reputação institucional e menor intervenção estatal.",
    "Liberais sociais": "combina defesa de liberdade individual, mercado, democracia institucional e certo distanciamento de polos partidários tradicionais.",
    "Empreendedores individuais": "valoriza esforço próprio, renda, autonomia, oportunidade e soluções práticas, mesmo convivendo com instabilidade e informalidade.",
    "Extrema direita": "apresenta visão antissistema, nacionalista, conservadora nos costumes e desconfiada de instituições, mídia e Estado.",
}

GENERATION_DESCRIPTIONS = {
    "Geração Bossa Nova": "formada em um Brasil mais analógico, familiar, religioso e orientado por autoridade e vizinhança.",
    "Geração Ordem e Progresso": "formada entre ditadura, hiperinflação, redemocratização e mudanças econômicas profundas.",
    "Geração Redemocratização": "formada na Nova República, expansão de direitos, estabilidade monetária e precarização do trabalho.",
    "Geração.Com": "nativa digital, marcada por smartphones, redes sociais, diversidade, pandemia e polarização política recente.",
}


def _other_race_to_allowed(value: str, rng: random.Random) -> str:
    if value != "Outra":
        return value
    return weighted_choice(rng, {"Amarela": 0.35, "Indígena": 0.45, "Prefere não responder": 0.20})


def _income_pc_to_family_income(value: str, rng: random.Random) -> str:
    mapping = {
        "Até 1/4 SM": {"Até 1 salário mínimo": 0.75, "Mais de 1 a 2 salários mínimos": 0.25},
        "Mais de 1/4 até 1/2 SM": {"Até 1 salário mínimo": 0.35, "Mais de 1 a 2 salários mínimos": 0.45, "Mais de 2 a 3 salários mínimos": 0.20},
        "Mais de 1/2 até 1 SM": {"Mais de 1 a 2 salários mínimos": 0.25, "Mais de 2 a 3 salários mínimos": 0.40, "Mais de 3 a 5 salários mínimos": 0.35},
        "Mais de 1 SM": {"Mais de 3 a 5 salários mínimos": 0.25, "Mais de 5 a 10 salários mínimos": 0.35, "Mais de 10 a 20 salários mínimos": 0.25, "Mais de 20 salários mínimos": 0.15},
    }
    return weighted_choice(rng, mapping[value])


def _location_from_territory(tipo_de_território: str) -> str:
    if tipo_de_território == "Capital":
        return "Capital"
    if tipo_de_território == "Cidade grande":
        return "Região metropolitana"
    if tipo_de_território == "Cidade média":
        return "Interior urbano"
    return "Interior urbano"


def _choose_uf(segment: str, config: dict[str, Any], rng: random.Random) -> str:
    weights = dict(config["uf_weights"])
    for uf in config.get("uf_overindex", {}).get(segment, []):
        if uf in weights:
            weights[uf] *= 1.75
    return weighted_choice(rng, weights)


def _choose_access_to_internet(renda_pc: str, geração: str, rng: random.Random) -> str:
    if geração == "Geração.Com":
        return weighted_choice(rng, {"Tem internet fixa em casa": 0.42, "Usa principalmente internet móvel": 0.52, "Usa internet fora de casa": 0.05, "Não tem acesso regular à internet": 0.01})
    if renda_pc in {"Até 1/4 SM", "Mais de 1/4 até 1/2 SM"}:
        return weighted_choice(rng, {"Tem internet fixa em casa": 0.25, "Usa principalmente internet móvel": 0.55, "Usa internet fora de casa": 0.15, "Não tem acesso regular à internet": 0.05})
    return weighted_choice(rng, {"Tem internet fixa em casa": 0.62, "Usa principalmente internet móvel": 0.34, "Usa internet fora de casa": 0.03, "Não tem acesso regular à internet": 0.01})


def _choose_housing(renda_pc: str, tipo_de_território: str, rng: random.Random) -> tuple[str, str]:
    if tipo_de_território == "Capital":
        tipo = weighted_choice(rng, {"Apartamento": 0.45, "Casa": 0.38, "Casa em condomínio": 0.08, "Quarto / cômodo / pensão": 0.06, "Cortiço / habitação coletiva": 0.03})
    elif tipo_de_território == "Cidade pequena":
        tipo = weighted_choice(rng, {"Casa": 0.74, "Moradia rural": 0.14, "Apartamento": 0.06, "Casa em condomínio": 0.04, "Outro tipo de moradia": 0.02})
    else:
        tipo = weighted_choice(rng, {"Casa": 0.58, "Apartamento": 0.25, "Casa em condomínio": 0.09, "Quarto / cômodo / pensão": 0.04, "Moradia rural": 0.03, "Outro tipo de moradia": 0.01})

    if renda_pc in {"Até 1/4 SM", "Mais de 1/4 até 1/2 SM"}:
        cond = weighted_choice(rng, {"Imóvel alugado": 0.38, "Imóvel cedido por familiar ou terceiros": 0.25, "Imóvel próprio quitado": 0.20, "Ocupação / moradia sem pagamento formal": 0.12, "Imóvel próprio financiado": 0.05})
    else:
        cond = weighted_choice(rng, {"Imóvel próprio quitado": 0.38, "Imóvel financiado": 0.02}) if False else weighted_choice(rng, {"Imóvel próprio quitado": 0.38, "Imóvel próprio financiado": 0.22, "Imóvel alugado": 0.32, "Imóvel cedido por familiar ou terceiros": 0.07, "Ocupação / moradia sem pagamento formal": 0.01})
    return cond, tipo


def _qualitative_text(dimension: str, profile: dict[str, Any], config: dict[str, Any]) -> str:
    segment = profile["segmento_e_bolhas"]
    generation = profile["geração"]
    gender = profile["gênero"]
    race = profile["raça/cor"]
    income = profile["renda_pc"]
    political = profile["identificação política"]
    ideology = profile["ideologia"]
    segment_desc = SEGMENT_DESCRIPTIONS.get(segment, "combina valores e práticas heterogêneas.")
    generation_desc = GENERATION_DESCRIPTIONS.get(generation, "formada por experiências sociais variadas.")
    base = config["values_dimensions"].get(dimension, "Dimensão psicográfica do perfil.")

    if dimension == "Religiosidade":
        return f"{base} Neste perfil, a religiosidade é interpretada pela combinação entre {segment_desc} A trajetória geracional é {generation_desc} O recorte de {gender}, {race} e renda '{income}' ajusta o grau de centralidade prática da fé e sua relação com tolerância religiosa."
    if dimension == "Família":
        return f"{base} A família tende a funcionar como rede de proteção e orientação moral, mas com abertura variável a novos arranjos conforme {generation} e ideologia {ideology}. O segmento {segment} dá o tom principal dessa visão."
    if dimension == "Honestidade":
        return f"{base} O julgamento moral tende a oscilar entre valorização de regras, sobrevivência cotidiana e tolerância à informalidade. O perfil de {segment} e renda '{income}' ajuda a calibrar se o jeitinho aparece como criatividade, necessidade ou desvio moral."
    if dimension == "Meritocracia":
        return f"{base} A leitura sobre esforço pessoal e ajuda social é moldada por {segment}, renda '{income}' e geração {generation}. O perfil pode valorizar batalha individual sem necessariamente rejeitar todo tipo de proteção pública."
    if dimension == "Ideologia":
        return f"{base} A identidade política declarada é {political} e a autoimagem ideológica é {ideology}. Na prática, isso deve ser tratado como combinação entre costumes, Estado, segurança, economia e pertencimento de grupo, não como programa partidário perfeitamente coerente."
    if dimension == "Discriminação":
        return f"{base} A percepção sobre racismo, machismo e desigualdade deve considerar o lugar social de {gender}, {race}, renda '{income}' e geração {generation}. O segmento {segment} influencia a disposição a reconhecer desigualdades estruturais ou tratá-las como casos individuais."
    if dimension == "Sexualidade":
        return f"{base} A aceitação de diversidade sexual varia conforme geração e conservadorismo nos costumes. Como {generation_desc}, este perfil combina sua experiência geracional com as normas do segmento {segment}."
    if dimension == "Comportamento social":
        return f"{base} A vida cotidiana combina circulação no território {profile['tipo_de_território']}, hábitos de mídia e sensação de segurança. O acesso à internet '{profile['acesso à internet']}' é relevante para estimar exposição a redes, TV, portais e sociabilidade presencial."
    if dimension == "Identidade nacional":
        return f"{base} A identidade nacional tende a misturar orgulho do Brasil cultural/natural com crítica a corrupção, violência e desigualdade. O segmento {segment} e a UF {profile['uf']} ajudam a dar coloração regional e política a essa percepção."
    if dimension == "Insegurança":
        return f"{base} A sensação de insegurança é calibrada por território, gênero, raça e renda. Neste perfil, {gender}, {race}, renda '{income}' e moradia em {profile['tipo_de_território']} indicam como medo, proteção familiar e punitivismo podem aparecer."
    if dimension == "Confiança":
        return f"{base} A confiança tende a ser maior em família e redes próximas, e menor em desconhecidos ou pessoas politicamente distantes. O segmento {segment} define a principal bolha de confiança e mediação informacional."
    return base


def _persona_text(profile: dict[str, Any]) -> str:
    p1 = (
        f"Pessoa de {profile['idade']}, {profile['gênero']}, {profile['raça/cor']}, residente em {profile['uf']} "
        f"em {profile['tipo_de_território']}. Pertence à {profile['geração']} e ao segmento '{profile['segmento_e_bolhas']}'. "
        f"Tem renda familiar mensal '{profile['renda familiar mensal']}', escolaridade '{profile['escolaridade']}' "
        f"e situação ocupacional '{profile['situação ocupacional']}'. Sua orientação política declarada é '{profile['identificação política']}' "
        f"e sua autoimagem ideológica é '{profile['ideologia']}'."
    )
    p2 = (
        f"Como agente sintético, deve responder a estímulos considerando uma combinação de dados demográficos, valores e atitudes. "
        f"Sua visão de mundo combina {SEGMENT_DESCRIPTIONS.get(profile['segmento_e_bolhas'], 'experiências sociais variadas')} "
        f"e uma trajetória geracional {GENERATION_DESCRIPTIONS.get(profile['geração'], 'heterogênea')}. "
        f"Consome informação principalmente por: {profile['tipo_de_fontes_de_informação']}."
    )
    return p1 + "\n\n" + p2


def _generate_one(i: int, config: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    segment_weights = {k: v["population"] for k, v in config["segment_distributions"].items()}
    segment = weighted_choice(rng, segment_weights)
    seg_cfg = config["segment_distributions"][segment]

    generation = weighted_choice(rng, seg_cfg["geração"])
    age = weighted_choice(rng, AGE_BY_GENERATION[generation])
    gender = weighted_choice(rng, seg_cfg["gênero"])
    race = _other_race_to_allowed(weighted_choice(rng, seg_cfg["raça/cor"]), rng)
    renda_pc = weighted_choice(rng, seg_cfg["renda_pc"])
    family_income = _income_pc_to_family_income(renda_pc, rng)
    territory = weighted_choice(rng, seg_cfg["território"])
    uf = _choose_uf(segment, config, rng)

    education = weighted_choice(rng, EDUCATION_BY_SEGMENT.get(segment, {"Ensino médio completo": 0.50, "Ensino superior completo": 0.30, "Ensino fundamental completo": 0.20}))
    occupation = weighted_choice(rng, OCCUPATION_BY_SEGMENT.get(segment, {"Empregado com carteira assinada": 0.35, "Autônomo / conta própria": 0.18, "Funcionário público": 0.12, "Empregado sem carteira assinada": 0.10, "MEI": 0.08, "Estudante": 0.07, "Aposentado / pensionista": 0.10}))

    if age == "65 anos ou mais":
        occupation = weighted_choice(rng, {occupation: 0.25, "Aposentado / pensionista": 0.75})
    if age == "16-24 anos":
        occupation = weighted_choice(rng, {occupation: 0.45, "Estudante": 0.40, "Empregado sem carteira assinada": 0.15})

    political = weighted_choice(rng, POLITICS_BY_SEGMENT[segment])
    ideology = weighted_choice(rng, IDEOLOGY_BY_SEGMENT[segment])
    cond_moradia, tipo_moradia = _choose_housing(renda_pc, territory, rng)
    access = _choose_access_to_internet(renda_pc, generation, rng)
    fontes = "; ".join(seg_cfg["fontes"])

    profile: dict[str, Any] = {
        "idade": age,
        "gênero": gender,
        "escolaridade": education,
        "situação ocupacional": occupation,
        "estado civil": weighted_choice(rng, {"Solteiro(a)": 0.34, "Casado(a)": 0.36, "União estável": 0.12, "Divorciado(a)": 0.09, "Separado(a)": 0.03, "Viúvo(a)": 0.06}),
        "condição de moradia": cond_moradia,
        "tipo de moradia": tipo_moradia,
        "localização": _location_from_territory(territory),
        "renda familiar mensal": family_income,
        "renda_pc": renda_pc,
        "acesso à internet": access,
        "tamanho do domicílio": weighted_choice(rng, {"1 pessoa": 0.15, "2-3 pessoas": 0.45, "4-5 pessoas": 0.30, "6 pessoas ou mais": 0.10}),
        "telefone": weighted_choice(rng, {"Apenas celular": 0.72, "Celular e telefone fixo": 0.24, "Apenas telefone fixo": 0.02, "Não possui telefone": 0.02}),
        "identificação política": political,
        "ideologia": ideology,
        "raça/cor": race,
        "geração": generation,
        "segmento_e_bolhas": segment,
        "tipo_de_território": territory,
        "uf": uf,
        "tipo_de_fontes_de_informação": fontes,
    }
    for dimension in config["values_dimensions"].keys():
        profile[dimension if dimension not in {"Comportamento social", "Identidade nacional"} else dimension.replace(" ", "_")] = _qualitative_text(dimension, profile, config)

    profile["Persona"] = _persona_text(profile)
    profile["id_persona"] = deterministic_id("agent", {"i": i, **profile}, digits=12)
    return profile


def generate_profiles(
    n: int,
    output_csv: str | Path | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a population-weighted synthetic audience base.

    The generator starts with the segment distribution, then samples conditional
    distributions for generation, gender, race, income and territory. The output is
    a survey-like table: one row per synthetic respondent/agent.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    config = load_config(config_path)
    rng = random.Random(seed)
    rows = [_generate_one(i=i, config=config, rng=rng) for i in range(n)]
    df = pd.DataFrame(rows)
    # Keep id first and remove internal helper income used in persona construction.
    cols = ["id_persona"] + [c for c in df.columns if c not in {"id_persona", "renda_pc"}] + ["renda_pc"]
    df = df[cols]
    if output_csv:
        out = ensure_parent(output_csv)
        df.to_csv(out, index=False, encoding="utf-8-sig")
    return df
