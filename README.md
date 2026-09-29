# Synthetic Audiences ABM

MVP em Python para criar uma base de **synthetic audiences** com agentes populacionais e aplicar questionários a artefatos de comunicação.

O projeto foi desenhado para ser uma primeira versão de demonstração para o Aberje Trends, mas já com estrutura de repositório escalável para evoluir para produto, pesquisa acadêmica e experimentos de Agent-Based Modeling + Cognitive Modeling.

## O que este MVP faz

1. **Gera perfis sintéticos populacionais** em formato de survey (`.csv`).
2. Usa pesos de referência para **Segmentos e Bolhas**, geração, gênero, raça/cor, renda, política e território.
3. Preenche variáveis:
   - `Demographic-Based`
   - `Survey-Based`
   - `Persona-Based`
4. Cria um `id_persona` único para cada agente.
5. Gera prompts do agente:
   - `prompt_user`
   - `prompt_system`
   - `prompt_reasoning`
6. Aplica questionários a um artefato, por exemplo texto de notícia.
7. Roda cada pergunta várias vezes por agente e calcula média/desvio padrão para respostas numéricas.
8. Salva resultados por agente, pergunta, iteração e artefato.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
```

Para usar OpenAI:

```bash
cp .env.example .env
# edite OPENAI_API_KEY e OPENAI_MODEL
```

## Uso rápido sem gastar API

Gerar 100 agentes:

```bash
synthaud generate --n 100 --output data/output/agents.csv
```

Aplicar questionário de exemplo em modo mock:

```bash
synthaud run \
  --agents-csv data/output/agents.csv \
  --questions-csv examples/questionnaire_example.csv \
  --artifact-json examples/artifact_example.json \
  --repeats 3 \
  --mock
```

Saídas:

- `data/output/agents.csv`: base dos agentes sintéticos.
- `data/output/responses.jsonl`: uma linha por agente × pergunta × iteração.
- `data/output/summary.csv`: média, desvio padrão, n e confiança média por pergunta.

## Uso com OpenAI

```bash
synthaud run \
  --agents-csv data/output/agents.csv \
  --questions-csv examples/questionnaire_example.csv \
  --artifact-json examples/artifact_example.json \
  --repeats 3
```

O wrapper tenta usar a Responses API e, se necessário, faz fallback para Chat Completions com JSON mode.

## Estrutura do repositório

```text
synthetic-audiences-abm/
├── data/
│   ├── input/
│   ├── output/
│   └── reference/
│       └── reference_config.yml
├── docs/
│   └── reference/
│       ├── 2_Model-Based_Agents_building.md
│       ├── 3_Population-level.md
│       └── 4_Structural_project.md
├── examples/
│   ├── artifact_example.json
│   └── questionnaire_example.csv
├── src/
│   └── synthetic_audiences/
│       ├── population.py
│       ├── prompts.py
│       ├── llm.py
│       ├── runner.py
│       ├── cli.py
│       └── langgraph_app.py
└── tests/
```

## Como evoluir depois do MVP

- Trocar os pesos YAML por tabelas versionadas vindas de IBGE, surveys, bases proprietárias e pesquisas qualitativas.
- Adicionar validação por distribuição populacional e calibração contra benchmarks reais.
- Criar múltiplos modos de agente: demographic-only, survey-based, persona-based e hybrid.
- Adicionar memória/`Dialogue history` via arquivos, banco vetorial ou checkpoints.
- Usar LangGraph para orquestrar nós de prompt, resposta, crítica, validação, retry e agregação.
- Rodar experimentos com estabilidade: 3–5 respostas por agente, variações de prompt e análise de sensibilidade.
- Criar avaliação com ground truth: survey real, painel, voto, recall de marca, confiança, intenção de compra ou reputação.

## Aviso metodológico

Este projeto não deve ser tratado como substituto direto de amostras humanas reais. A proposta é criar um laboratório de simulação e prototipagem para hipóteses de comunicação, reputação e comportamento, com validação progressiva contra dados reais.
