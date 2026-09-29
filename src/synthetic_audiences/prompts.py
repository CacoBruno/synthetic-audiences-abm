from __future__ import annotations

import json
from typing import Any

import pandas as pd

PROFILE_COLUMNS = [
    "id_persona",
    "idade",
    "gênero",
    "raça/cor",
    "uf",
    "tipo_de_território",
    "geração",
    "segmento_e_bolhas",
    "escolaridade",
    "renda familiar mensal",
    "situação ocupacional",
    "identificação política",
    "ideologia",
    "tipo_de_fontes_de_informação",
    "Persona",
]


def build_prompt_user(agent_profile: dict[str, Any] | pd.Series) -> str:
    """Build the profile/persona prompt for an agent."""
    if isinstance(agent_profile, pd.Series):
        profile = agent_profile.to_dict()
    else:
        profile = dict(agent_profile)

    selected = {k: profile.get(k) for k in PROFILE_COLUMNS if k in profile}
    qualitative = {
        key: profile.get(key)
        for key in [
            "Religiosidade",
            "Família",
            "Honestidade",
            "Meritocracia",
            "Ideologia",
            "Discriminação",
            "Sexualidade",
            "Comportamento_social",
            "Identidade_nacional",
            "Insegurança",
            "Confiança",
        ]
        if key in profile
    }
    return (
        "Você está simulando um respondente sintético brasileiro.\n"
        "Use apenas as informações abaixo como base do perfil; não invente biografias específicas, nomes, eventos pessoais ou histórico familiar.\n\n"
        "## Perfil estruturado\n"
        f"{json.dumps(selected, ensure_ascii=False, indent=2)}\n\n"
        "## Valores, atitudes e percepções\n"
        f"{json.dumps(qualitative, ensure_ascii=False, indent=2)}"
    )


def build_prompt_system(task: str | None = None) -> str:
    """Build system-level instructions for the synthetic respondent."""
    task_text = task or "Responder a perguntas de pesquisa sobre um artefato de comunicação."
    return (
        "Você é um agente de pesquisa para simulação de audiência sintética. "
        "Seu papel é prever respostas prováveis de um perfil populacional, não agir como uma pessoa real. "
        "Priorize coerência entre demografia, segmento, geração, território, valores e consumo de informação. "
        "Evite estereótipos excessivos: quando a informação for insuficiente, reduza a confiança. "
        "Responda sempre em JSON válido, sem markdown.\n\n"
        f"Tarefa: {task_text}"
    )


def build_prompt_reasoning(mode: str = "thinking", dialogue_history: str | None = None) -> str:
    """Build reasoning instructions without asking the model to expose chain-of-thought.

    Dialogue history is kept as a placeholder for future memory. Thinking mode asks the
    model to deliberate internally but return only answer, rationale and confidence.
    """
    history = dialogue_history or "Nenhum histórico de diálogo disponível nesta versão."
    if mode == "none":
        reasoning = "Não faça raciocínio longo; responda diretamente com uma justificativa curta."
    else:
        reasoning = (
            "Antes de responder, avalie internamente: perfil, segmento, geração, valores, pergunta, artefato e escala. "
            "Não revele cadeia de pensamento. Retorne apenas uma justificativa sintética e auditável."
        )
    return f"## Dialogue history\n{history}\n\n## Thinking mode\n{reasoning}"


def build_question_prompt(question: dict[str, Any], artifact: dict[str, Any]) -> str:
    """Build the questionnaire prompt for one question and one artifact."""
    options = question.get("options")
    options_text = ""
    if isinstance(options, list) and options:
        options_text = "\nOpções: " + "; ".join(str(o) for o in options)
    return (
        "Avalie o artefato abaixo e responda à pergunta como o perfil sintético responderia.\n\n"
        f"## Artefato\nID: {artifact.get('artifact_id')}\nTipo: {artifact.get('artifact_type', 'news_text')}\nConteúdo:\n{artifact.get('content')}\n\n"
        f"## Pergunta\nID: {question.get('question_id')}\nConstruct: {question.get('construct')}\nTexto: {question.get('question_text')}\n"
        f"Tipo de resposta: {question.get('response_type', 'likert_1_5')}{options_text}\n\n"
        "Retorne JSON com as chaves: answer, score_numeric, rationale, confidence. "
        "Para likert_1_5, answer deve ser inteiro de 1 a 5 e score_numeric deve repetir esse número. "
        "Para single_choice, answer deve ser uma das opções e score_numeric deve ser null."
    )


def build_prompt_agent(
    agent_profile: dict[str, Any] | pd.Series,
    question: dict[str, Any],
    artifact: dict[str, Any],
    task: str | None = None,
    reasoning_mode: str = "thinking",
    dialogue_history: str | None = None,
) -> list[dict[str, str]]:
    """Build OpenAI-compatible messages: system + user."""
    system = build_prompt_system(task) + "\n\n" + build_prompt_reasoning(reasoning_mode, dialogue_history)
    user = build_prompt_user(agent_profile) + "\n\n" + build_question_prompt(question, artifact)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
