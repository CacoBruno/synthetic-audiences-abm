import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.synthetic_audiences.agents.context import build_agent_context, safe_get
from src.synthetic_audiences.artifacts.news import check_source_familiarity
from src.synthetic_audiences.settings import DEFAULT_MODEL, DEFAULT_TEMPERATURE


def build_llm(model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY não encontrada.")
    return ChatOpenAI(model=model, temperature=temperature)


def scale_to_text(question: dict) -> str:
    return "\n".join(
        f"{k}: {v}" for k, v in question["scale_labels"].items()
    )


def build_agent_question_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Você é um simulador de respondentes sintéticos para um projeto de Synthetic Audiences baseado em Agent-Based Modeling e Cognitive Modeling.

Você deve responder como se fosse uma pessoa real participando de uma pesquisa.
Você não deve responder como especialista, consultor, pesquisador ou modelo de IA.

REGRAS GERAIS:
1. Leia primeiro a Persona.
2. Depois use as variáveis socio-demográficas.
3. Depois use as variáveis de valores, atitudes e percepções.
4. Responda como esse agente tenderia a responder.
5. Use a escala exatamente como apresentada.
6. Retorne apenas JSON válido.
7. Não exponha raciocínio passo a passo.
8. A justificativa deve ter no máximo duas frases.

FORMATO OBRIGATÓRIO DO JSON:
{{
  "answer_value": número inteiro dentro da escala,
  "answer_label": "rótulo exato da escala",
  "construct": "nome do constructo",
  "rationale": "justificativa curta em português",
  "confidence": número entre 0 e 1,
  "source_familiarity_effect": "como a familiaridade com a fonte influenciou a resposta",
  "agent_variable_reading": {{
    "persona_summary": "síntese curta de como a persona influenciou a resposta",
    "socio_demographic_summary": "síntese curta de como as variáveis socio-demográficas influenciaram a resposta",
    "values_attitudes_perceptions_summary": "síntese curta de como valores, atitudes e percepções influenciaram a resposta"
  }}
}}
""".strip(),
            ),
            (
                "human",
                """
CONTEXTO DO AGENTE

{agent_context}

CONTEXTO DA FONTE DA NOTÍCIA

Tipo de fonte da notícia:
{artifact_source}

Fontes que o agente costuma consumir:
{agent_sources}

A fonte da notícia é familiar para o agente?
{fonte_familiar}

Orientação sobre familiaridade com a fonte:
{source_alignment_instruction}

ARTEFATO ANALISADO

ID do artefato:
{artifact_id}

Tipo do artefato:
{artifact_type}

Título:
{artifact_title}

Conteúdo:
{artifact_content}

APLICAÇÃO DO QUESTIONÁRIO

Agora você será exposto a uma pergunta de questionário, como em uma pesquisa aplicada a seres humanos.

Instruções ao respondente:
- Leia a notícia.
- Responda à pergunta pensando na sua impressão pessoal.
- Não tente dar uma resposta tecnicamente correta.
- Não tente agradar o pesquisador.
- Responda de acordo com seus valores, experiências, hábitos de mídia e visão de mundo.
- Use apenas uma opção da escala.

Constructo medido:
{construct}

Pergunta:
{question}

Instrução específica de resposta:
{response_instruction}

Escala:
{scale_text}

Retorne apenas JSON válido.
""".strip(),
            ),
        ]
    )


def build_agent_chain(llm=None):
    llm = llm or build_llm()
    prompt = build_agent_question_prompt()
    parser = JsonOutputParser()
    return prompt | llm | parser


def run_agent_question(
    agent_row: pd.Series,
    question: dict,
    artifact: dict,
    chain=None,
) -> dict:
    chain = chain or build_agent_chain()

    source_info = check_source_familiarity(
        agent_sources=safe_get(agent_row, "tipo_de_fontes_de_informação", ""),
        artifact_source=artifact.get("tipo_de_fontes_de_informação", ""),
    )

    result = chain.invoke(
        {
            "agent_context": build_agent_context(agent_row),
            "artifact_source": artifact.get("tipo_de_fontes_de_informação", ""),
            "agent_sources": source_info["agent_sources"],
            "fonte_familiar": source_info["fonte_familiar"],
            "source_alignment_instruction": source_info["source_alignment_instruction"],
            "artifact_id": artifact.get("artifact_id", ""),
            "artifact_type": artifact.get("artifact_type", ""),
            "artifact_title": artifact.get("title", ""),
            "artifact_content": artifact.get("content", ""),
            "construct": question["construct"],
            "question": question["question"],
            "response_instruction": question["response_instruction"],
            "scale_text": scale_to_text(question),
        }
    )

    result["answer_value"] = int(result["answer_value"])
    result["confidence"] = float(result.get("confidence", 0.75))
    result["fonte_familiar"] = source_info["fonte_familiar"]
    result["artifact_source"] = source_info["artifact_source"]
    result["agent_sources"] = source_info["agent_sources"]

    return result
