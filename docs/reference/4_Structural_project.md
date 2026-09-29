

a ideal é agora fazer um estudo, como uma teste, mas será estruturado como um projeto para vira um produto escalável; agora eu preciso apenas das funções de geração dos agentes e de aplicação dos agentes;

Mas precisa ser pensado já estruturalmente para se escalável e salvo em um repositório do Github

### geração dos agentes
a função tem de conseguir ler as tabelas e preencher de acordo com os perfis estabelecidos
a ideia é que o output seja similar ao perfil de uma base de pesquisa população (como em um survey); as informações de como são os conceitos, as tabelas e as variáveis a serem criadas estão em [[2. Model-Based Agents building]]

1. função que gera os perfis ([[3. Population-level]])
	1. quantidade de perfis
	2. calculo do perfil populacional dos agentes
		1. tabelas sobre o perfil da população
			1. demográfico
			2. survey
	3. aplicação das variáveis demográficas (Demographic-based)
		1. tem ser gerado a partir do calculo dos perfil demográfico
		2. leitura das tabelas de variáveis para aplicação (essas tabelas estão em [[2. Model-Based Agents building]] a partir da seção "informações qualitativas para o preenchimento das variáveis")
	4. aplicação das variáveis da pesquisa de perfil (Survey-based)
		1. tem de gerado a partir do perfil gerado pela survey
		2. leitura de cruzamentos da tabela de survey
	5. geração de texto sober a persona do agent (Persona-based)
	6. geração de um id único para cada agent
	7. salva em uma base de dados (agora, será um .csv)
### aplicação dos agentes 

1. base de dados  (.csv) com os perfis de cada agente gerado pelas funções de geração de agents
2. função que gera o prompt (prompt_user) dos perfis a partir da base de dados
3. função que gera a ação que será feita (prompt_system)
4. função que gera o reasoning (prompt_reasoning)
	1. **Dialogue history**: (não precisa ser criado agora; será aplicado como um tipo de memória -> o arquivo pode ser gerado em branco)
	2. **Thinking mode**
5. função que gera o agente (prompt_agent: prompt_user + prompt_system + prompt_reasoning)
6. função que lê o teste (precisa padronizar para inserir novos) aplica para os agentes os questionários 
	1. input: tabelas com os questionários
	2. input: dado a ser analisado; será o id do artefato analisado e o artefato (agora será texto de notícias, mas depois imagem, vídeo, etc...)
	3. output .json será o construct e as suas respostas
	4.  o ideal seria que a função fosse aplicada 3-5 vez de uma vez e a resposta fosse a média das respostas; trazendo também o desvio padrão (teste: se for sempre igual podemos tirar essa função)
	5. no final tem de ser salvo com a repostas para cada reposta por agente; com os ids de cada agente e id de cada artefato 



