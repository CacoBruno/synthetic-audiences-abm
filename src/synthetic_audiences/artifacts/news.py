def build_news_artifact(
    artifact_id: str,
    title: str,
    content: str,
    tipo_de_fontes_de_informação: str,
    source_name: str = "",
    artifact_type: str = "news_text",
) -> dict:
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "title": title,
        "content": content,
        "tipo_de_fontes_de_informação": tipo_de_fontes_de_informação,
        "source_name": source_name,
    }


def check_source_familiarity(agent_sources: str, artifact_source: str) -> dict:
    agent_sources = agent_sources or ""
    artifact_source = artifact_source or ""

    is_familiar = artifact_source.lower() in agent_sources.lower()

    if is_familiar:
        instruction = """
O tipo de fonte desta notícia é familiar para o agente.
Como o agente costuma consumir esse tipo de fonte, ele tende a interpretar o conteúdo com maior familiaridade, maior abertura inicial e uma leitura relativamente mais otimista.
Essa orientação não deve forçar uma resposta positiva; ela apenas aumenta a predisposição de confiança e receptividade, desde que o conteúdo seja compatível com o perfil.
""".strip()
    else:
        instruction = """
O tipo de fonte desta notícia não é familiar para o agente.
Como o agente não costuma consumir esse tipo de fonte, ele tende a interpretar o conteúdo com maior distância, ceticismo e uma leitura relativamente mais pessimista.
Essa orientação não deve forçar uma resposta negativa; ela apenas aumenta a predisposição de cautela, dúvida ou menor confiança, de acordo com o perfil.
""".strip()

    return {
        "fonte_familiar": is_familiar,
        "agent_sources": agent_sources,
        "artifact_source": artifact_source,
        "source_alignment_instruction": instruction,
    }
