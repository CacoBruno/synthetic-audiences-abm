# Experimento Itaú — Crédito e Endividamento

Esta pasta implementa o primeiro teste de narrativas do projeto de audiências sintéticas aplicado ao estudo do Itaú.

## Perguntas de negócio

### 1. IA e o preço do crédito

O teste separa três interpretações que não devem ser tratadas como equivalentes:

- precificação dinâmica por condições de mercado;
- precificação personalizada baseada em características/comportamento do consumidor;
- precificação de crédito baseada em risco de inadimplência.

A pergunta experimental é se a narrativa do Itaú faz a IA ser entendida principalmente como instrumento de adequação à capacidade de pagamento e prevenção do endividamento ou se ativa a percepção de exploração da disposição a pagar, falta de transparência ou discriminação.

### 2. Fintechs e excesso de oferta

O contexto apresenta a associação observada entre expansão do crédito digital, entrada de novos usuários e crescimento da inadimplência sem afirmar causalidade.

A pergunta experimental é se o posicionamento do Itaú é percebido como crédito responsável e relacionamento de longo prazo ou como exclusão, paternalismo e ataque oportunista aos bancos digitais.

## Desenho experimental

Cada tema tem três condições independentes:

1. **context** — apenas o enquadramento crítico;
2. **narrative** — apenas a narrativa do Itaú;
3. **context_plus_narrative** — contexto crítico seguido da narrativa.

As chamadas são stateless: o mesmo agente pode aparecer nas três condições sem receber memória das exposições anteriores.

O desenho passa a separar duas funções:

- **Context** é diagnóstico do problema e usa perguntas próprias, sem atribuir ao Itaú uma posição que não apareceu no estímulo.
- **Resiliência da narrativa** = score(contexto + narrativa) − score(narrativa isolada).

Resiliência negativa significa perda de desempenho quando a controvérsia é conhecida; positiva significa manutenção ou ganho. O contexto isolado não entra em um score agregado de "lift" da marca porque o conjunto de perguntas não é o mesmo.

## Perfis financeiros

Os perfis não substituem `segmento_e_bolhas`. Eles são overlays sobre a população sintética já existente:

- endividado de baixa renda;
- classe média com renda comprometida;
- aposentado com consignado;
- jovem em primeiro crédito e cliente de banco digital;
- MEI;
- formador de opinião, dividido em jornalista econômico e educador financeiro;
- cliente de alta renda.

Isso permite, por exemplo, dois jovens de primeiro crédito com segmentos sociopolíticos e visões de mundo diferentes.

## Modelo

O experimento usa por padrão:

```text
gpt-6-luna
reasoning.effort = low
```

O Luna é usado porque o workload é estruturado, repetitivo e de alto volume. Cada agente responde todo o bloco de itens de uma condição em **uma única chamada**.

## 1. Construir os agentes experimentais

Na raiz do projeto:

```powershell
python experiments/itau_credit_narratives/build_agents.py
```

Por padrão, o script gera primeiro uma população-base reproduzível de **5.000 personas individuais** usando `data/reference/reference_config.yml` e `seed=42`. Em seguida, seleciona somente personas elegíveis para cada perfil financeiro e aplica o overlay experimental.

O padrão é 10 agentes por perfil, totalizando 70. Não existe fallback genérico: se algum perfil não tiver candidatos suficientes, o pipeline interrompe a execução.

Para alterar:

```powershell
python experiments/itau_credit_narratives/build_agents.py --n-per-profile 30 --base-population-size 10000
```

Também é possível fornecer uma base própria de personas individuais:

```powershell
python experiments/itau_credit_narratives/build_agents.py --source caminho/agents.csv
```

O arquivo `cluster_perfil_persona.csv` permanece versionado no projeto como base de resultados agregados do pipeline anterior. Ele **não é usado para selecionar estes perfis financeiros**, porque não contém as variáveis individuais necessárias, como idade, renda e ocupação.

Saída:

```text
experiments/itau_credit_narratives/outputs/agents_itau_credit.csv
```

## 2. Smoke test sem API

Use primeiro:

```powershell
python experiments/itau_credit_narratives/run_experiment.py --mock --limit-agents 7 --repeats 1
```

Isso valida schema, persistência e análise sem gastar API. O limite de 7 agentes é amostrado de forma balanceada: com os sete perfis existentes, entra 1 agente de cada perfil.

## 3. Rodar com OpenAI

Configure o ambiente:

```powershell
$env:OPENAI_API_KEY="SUA_CHAVE"
$env:OPENAI_MODEL="gpt-6-luna"
$env:OPENAI_REASONING_EFFORT="low"
```

Depois:

```powershell
python experiments/itau_credit_narratives/run_experiment.py
```

O desenho padrão executa:

```text
70 agentes × 6 estímulos × 2 repetições = 840 chamadas
```

Cada chamada retorna todos os itens quantitativos, a categoria de interpretação, resposta aberta, crítica principal e confiança.

Você pode rodar apenas um tema:

```powershell
python experiments/itau_credit_narratives/run_experiment.py --theme ia_preco_credito
```

ou:

```powershell
python experiments/itau_credit_narratives/run_experiment.py --theme fintech_excesso_oferta
```

## 4. Analisar

```powershell
python experiments/itau_credit_narratives/analyze_results.py
```

Se `--run-dir` não for informado, o script analisa o run mais recente.

Principais saídas:

- `item_summary.csv`
- `construct_summary.csv`
- `overall_summary.csv`
- `message_resilience.csv`
- `construct_comparison.csv`
- `context_diagnostics.csv`
- `interpretation_shares.csv`
- `open_responses.csv`
- `report.md`

## Scores

As respostas originais permanecem em `score_raw`, sempre de 1 a 7.

Para análises comparáveis, existe também `score_favorable`. Nos itens positivos ele é igual ao score bruto. Nos itens adversos — exploração, discriminação, exclusão, paternalismo e ataque a fintechs — a transformação é:

```text
score_favorable = 8 - score_raw
```

Assim, score favorável alto significa melhor recepção da narrativa sem perder a resposta original.

## Regra metodológica

Os resultados sintéticos devem ser usados como laboratório de hipóteses, comparação de narrativas e priorização de riscos. Eles não substituem pesquisa humana. O passo posterior é validar as medidas e os padrões principais contra benchmark humano.
