from pathlib import Path

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_REPEATS = 1

SOCIO_DEMOGRAPHIC_COLUMNS = [
    "idade",
    "gênero",
    "escolaridade",
    "situação ocupacional",
    "estado civil",
    "condição de moradia",
    "tipo de moradia",
    "localização",
    "renda familiar mensal",
    "acesso à internet",
    "tamanho do domicílio",
    "telefone",
]

VALUES_ATTITUDES_PERCEPTIONS_COLUMNS = [
    "identificação política",
    "ideologia",
    "raça/cor",
    "geração",
    "segmento_e_bolhas",
    "tipo_de_território",
    "uf",
    "tipo_de_fontes_de_informação",
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

DEFAULT_GROUP_VARIABLES = [
    "segmento_e_bolhas",
    "geração",
    "ideologia",
    "identificação política",
    "raça/cor",
    "gênero",
    "renda familiar mensal",
    "uf",
    "tipo_de_território",
    "tipo_de_fontes_de_informação",
    "fonte_familiar",
]
