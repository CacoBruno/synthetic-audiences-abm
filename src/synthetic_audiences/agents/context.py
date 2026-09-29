import pandas as pd
from src.synthetic_audiences.settings import (
    SOCIO_DEMOGRAPHIC_COLUMNS,
    VALUES_ATTITUDES_PERCEPTIONS_COLUMNS,
)

def safe_get(row: pd.Series, col: str, default: str = "") -> str:
    if col not in row.index:
        return default
    value = row[col]
    if pd.isna(value):
        return default
    return str(value)


def format_fields(row: pd.Series, columns: list[str]) -> str:
    return "\n".join(
        f"- {col}: {safe_get(row, col, 'Não informado')}"
        for col in columns
    )


def build_agent_context(row: pd.Series) -> str:
    """
    Constrói o contexto completo do agente em três blocos:
    1. Persona
    2. Variáveis socio-demográficas
    3. Valores, atitudes e percepções
    """
    id_persona = safe_get(row, "id_persona", "sem_id")
    persona = safe_get(row, "Persona", "Persona não informada.")
    socio = format_fields(row, SOCIO_DEMOGRAPHIC_COLUMNS)
    vap = format_fields(row, VALUES_ATTITUDES_PERCEPTIONS_COLUMNS)

    return f"""
ID DO AGENTE
- id_persona: {id_persona}

1. PERSONA

A variável Persona é a descrição narrativa principal do agente.
Ela sintetiza quem é essa pessoa, seu contexto social, seus hábitos, suas tensões e sua visão geral de mundo.
Use esta descrição como ponto de partida para interpretar todas as demais variáveis.

Persona:
{persona}

2. VARIÁVEIS SOCIO-DEMOGRÁFICAS

As variáveis socio-demográficas descrevem a posição social objetiva do agente.
Elas indicam idade, gênero, escolaridade, ocupação, estado civil, moradia, território, renda, acesso a meios de comunicação e composição domiciliar.
Essas variáveis devem orientar como o agente interpreta riscos, oportunidades, instituições, empresas, notícias e temas públicos.

{socio}

3. VALORES, ATITUDES E PERCEPÇÕES

As variáveis de pesquisa descrevem a visão de mundo do agente.
Elas incluem identificação política, ideologia, raça/cor, geração, segmento sociopolítico, território, UF, fontes de informação e dimensões psicográficas.
Essas variáveis devem orientar a resposta do agente em termos de crenças, confiança, insegurança, moralidade, família, religião, meritocracia, discriminação, sexualidade, identidade nacional e comportamento social.

{vap}
""".strip()
