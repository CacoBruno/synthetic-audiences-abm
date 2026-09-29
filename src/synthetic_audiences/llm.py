from __future__ import annotations

import json
import os
import re
from typing import Any

from synthetic_audiences.utils import stable_float_0_1


JSON_RESPONSE_SCHEMA = {
    "name": "synthetic_audience_response",
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": ["string", "number", "integer"]},
            "score_numeric": {"type": ["number", "integer", "null"]},
            "rationale": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["answer", "score_numeric", "rationale", "confidence"],
        "additionalProperties": True,
    },
}


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from an LLM response."""
    text = text.strip()
    if not text:
        raise ValueError("Empty LLM response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(match.group(0))


class OpenAIJsonClient:
    """Small wrapper around the OpenAI SDK with a JSON fallback parser.

    The code first tries the Responses API and falls back to Chat Completions for
    environments pinned to older SDK/API behavior.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None, temperature: float = 0.2):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Use --mock or create a .env file.")
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key)

    def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        # Preferred path: Responses API.
        try:
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                temperature=self.temperature,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": JSON_RESPONSE_SCHEMA["name"],
                        "schema": JSON_RESPONSE_SCHEMA["schema"],
                    }
                },
            )
            return extract_json(response.output_text)
        except Exception:
            # Compatibility path: Chat Completions with JSON object mode.
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            return extract_json(response.choices[0].message.content or "")


def mock_agent_response(
    agent_id: str,
    question: dict[str, Any],
    artifact: dict[str, Any],
    iteration: int = 0,
) -> dict[str, Any]:
    """Deterministic local response for tests and dry runs without API calls."""
    response_type = question.get("response_type", "likert_1_5")
    u = stable_float_0_1(agent_id, question.get("question_id"), artifact.get("artifact_id"), iteration)
    if response_type == "single_choice":
        options = question.get("options") or ["A", "B", "C", "D"]
        idx = min(int(u * len(options)), len(options) - 1)
        return {
            "answer": options[idx],
            "score_numeric": None,
            "rationale": "Resposta mock determinística para teste local sem chamada de LLM.",
            "confidence": 0.55 + (u * 0.25),
        }
    if response_type == "open_text":
        return {
            "answer": "Síntese mock da percepção provável do agente sobre o artefato.",
            "score_numeric": None,
            "rationale": "Resposta textual mock para validação do pipeline.",
            "confidence": 0.55 + (u * 0.25),
        }
    score = int(u * 5) + 1
    score = max(1, min(5, score))
    return {
        "answer": score,
        "score_numeric": float(score),
        "rationale": "Pontuação mock determinística para validar média, desvio padrão e persistência.",
        "confidence": 0.55 + (u * 0.25),
    }
