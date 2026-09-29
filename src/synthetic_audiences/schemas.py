from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Likert5 = Literal[1, 2, 3, 4, 5]


class AgentProfile(BaseModel):
    id_persona: str
    idade: str
    gênero: str
    escolaridade: str
    situação_ocupacional: str = Field(alias="situação ocupacional")
    estado_civil: str = Field(alias="estado civil")
    condição_de_moradia: str = Field(alias="condição de moradia")
    tipo_de_moradia: str = Field(alias="tipo de moradia")
    localização: str
    renda_familiar_mensal: str = Field(alias="renda familiar mensal")
    acesso_à_internet: str = Field(alias="acesso à internet")
    tamanho_do_domicílio: str = Field(alias="tamanho do domicílio")
    telefone: str
    identificação_política: str = Field(alias="identificação política")
    ideologia: str
    raça_cor: str = Field(alias="raça/cor")
    geração: str
    segmento_e_bolhas: str
    tipo_de_território: str
    uf: str
    tipo_de_fontes_de_informação: str
    Religiosidade: str
    Família: str
    Honestidade: str
    Meritocracia: str
    Ideologia: str
    Discriminação: str
    Sexualidade: str
    Comportamento_social: str
    Identidade_nacional: str
    Insegurança: str
    Confiança: str
    Persona: str

    model_config = {"populate_by_name": True}


class Question(BaseModel):
    question_id: str
    construct_name: str = Field(alias="construct")
    question_text: str
    response_type: Literal["likert_1_5", "single_choice", "open_text"] = "likert_1_5"
    options: list[str] | None = None


class Artifact(BaseModel):
    artifact_id: str
    artifact_type: str = "news_text"
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunResponse(BaseModel):
    agent_id: str
    artifact_id: str
    question_id: str
    construct_name: str = Field(alias="construct")
    iteration: int
    response_type: str
    answer: str | int | float
    score_numeric: float | None = None
    rationale: str
    confidence: float = Field(ge=0, le=1)
    raw_response: dict[str, Any] = Field(default_factory=dict)
