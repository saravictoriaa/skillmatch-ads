import pandas as pd
from pathlib import Path

# CONFIGURAÇÃO DE CAMINHOS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CAMINHO_BRUTO = DATA_DIR / "survey_results_public.csv"
CAMINHO_SAIDA = DATA_DIR / "perfis_mercado.CSV"

# CONFIGURAÇÃO DAS COLUNAS
COLUNAS_INTERESSE = [
    "DevType",
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "PlatformHaveWorkedWith",
    "SOTagsHaveWorkedWith",
    "AIModelsHaveWorkedWith",
]

# CONFIGURAÇÃO DAS TRILHAS PROFISSIONAIS
TRILHAS_ALVO = {
    "Desenvolvimento Back-End e APIs":
        "Developer, back-end",

    "Desenvolvimento Front-End":
        "Developer, front-end",

    "Desenvolvimento Full-Stack":
        "Developer, full-stack",

    "Engenharia de Dados e IA":
        "Data engineer|AI/ML engineer|Data scientist",

    "Desenvolvimento Mobile":
        "Developer, mobile",

    "DevOps e Infraestrutura":
        "DevOps engineer or professional",

    "Segurança da Informação (Cybersec)":
        "Cybersecurity or InfoSec professional",

    "Qualidade de Software (QA)":
        "Developer, QA or test",
}

# PESOS DAS CATEGORIAS
PESOS_CATEGORIAS = {
    "LanguageHaveWorkedWith": 1.0,
    "DatabaseHaveWorkedWith": 1.0,
    "PlatformHaveWorkedWith": 1.3,
    "SOTagsHaveWorkedWith": 1.5,
}


# FUNÇÃO PARA EXTRAÇÃO DE VALORES
def extrair_valores_coluna(
    subset: pd.DataFrame,
    coluna: str
) -> list[str]:
    """
    Extrai valores separados por ';' de uma coluna
    e remove valores vazios.
    """

    if coluna not in subset.columns:
        return []

    valores = (
        subset[coluna]
        .fillna("")
        .astype(str)
        .str.split(";")
        .explode()
        .str.strip()
    )

    return [
        valor
        for valor in valores
        if valor
    ]

def calcular_skills_ponderadas(
    subset: pd.DataFrame,
    nome_trilha: str
) -> dict[str, float]:
    """
    Calcula a pontuação das competências de uma trilha com 
    base na frequência (ponderada por categoria).
    """

    categorias = PESOS_CATEGORIAS.copy()

    # Modelos de IA são específicos da trilha de IA/Dados.
    if nome_trilha == "Engenharia de Dados e IA":
        categorias["AIModelsHaveWorkedWith"] = 1.5

    pontuacoes = {}

    for coluna, peso in categorias.items():
        valores = extrair_valores_coluna(subset, coluna)
        for skill in valores:
            pontuacoes[skill] = pontuacoes.get(skill, 0.0) + peso

    return pontuacoes


# 1. EXTRAÇÃO
print("[ETL] Iniciando processamento da base pública oficial...")

if not CAMINHO_BRUTO.exists():
    print("[Erro crítico] Arquivo bruto não encontrado em:")
    print(CAMINHO_BRUTO)
    print("Certifique-se de que 'survey_results_public.csv' está dentro da pasta 'data/'.")
    raise SystemExit(1)

print("[Extração] Carregando arquivo bruto (isso pode levar alguns segundos)...")

df_raw = pd.read_csv(
    CAMINHO_BRUTO,
    usecols=COLUNAS_INTERESSE,
    low_memory=False
)

print(f"[Transformação] Total de registros brutos analisados: {len(df_raw)}")

# 2. IDENTIFICAÇÃO DOS PERFIS
perfis_pontuacoes = {}
quantidade_profissionais = {}

for nome_trilha, filtro_devtype in TRILHAS_ALVO.items():

    subset = df_raw[
        df_raw["DevType"]
        .fillna("")
        .str.contains(filtro_devtype, case=False, na=False, regex=True)
    ]

    quantidade_profissionais[nome_trilha] = len(subset)
    print(f"[ETL] {nome_trilha}: {len(subset)} profissionais encontrados")

    if subset.empty:
        print(f"[AVISO] Nenhum registro encontrado para '{filtro_devtype}'")
        perfis_pontuacoes[nome_trilha] = {}
        continue

    perfis_pontuacoes[nome_trilha] = calcular_skills_ponderadas(subset, nome_trilha)

# 3. SELEÇÃO DAS TOP SKILLS (Aumentado para 20 para maior precisão)
perfis_processados = []

for nome_trilha in TRILHAS_ALVO.keys():
    pontuacoes = perfis_pontuacoes[nome_trilha]

    skills_ordenadas = sorted(
        pontuacoes.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Pegando as top 20 skills para compor um perfil mais robusto (maior recall)
    top_skills = [skill for skill, _ in skills_ordenadas[:20]]
    skills_str = ", ".join(top_skills)

    # Mantendo nome das colunas "Trilha" e "Skills" para integração correta com app.py
    perfis_processados.append({
        "Trilha": nome_trilha,
        "Skills": skills_str,
    })

# 4. CARGA
df_final = pd.DataFrame(perfis_processados)

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Exportar em UTF-8 para evitar problemas de caracteres e manter como padrão
df_final.to_csv(
    CAMINHO_SAIDA,
    index=False,
    sep=";",
    encoding="utf-8"
)

# 5. VALIDAÇÃO FINAL
print("\n" + "=" * 70)
print("RESUMO DA BASE DE MERCADO GERADA")
print("=" * 70)

for _, row in df_final.iterrows():
    skills = [skill.strip() for skill in str(row["Skills"]).split(",") if skill.strip()]
    nome_trilha = row["Trilha"]

    print(f"\n{nome_trilha}")
    print(f"  Profissionais: {quantidade_profissionais[nome_trilha]}")
    print(f"  Skills selecionadas: {len(skills)}")
    print(f"  Top skills: {', '.join(skills)}")

print("\n[Carga] Sucesso! Base de mercado gerada em:")
print(CAMINHO_SAIDA)