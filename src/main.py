import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Ingestão de Dados (Apenas Base da Stack Overflow)
try:
    df_perfis = pd.read_csv('../data/perfis_mercado.CSV', encoding='utf-8', sep=';')
except FileNotFoundError as e:
    print(f"[Erro] Arquivo não encontrado: {e}")
    print("Por favor, execute 'python etl_survey.py' primeiro para gerar 'perfis_mercado.CSV'.")
    exit()

# Validação rápida de colunas
if not {'Trilha', 'Skills'}.issubset(df_perfis.columns):
    print("[Erro] O arquivo 'perfis_mercado.CSV' não contém as colunas 'Trilha' e 'Skills'.")
    exit()

df_perfis['Trilha'] = df_perfis['Trilha'].astype(str).str.strip()
df_perfis['Skills'] = df_perfis['Skills'].astype(str).str.strip()
df_perfis = df_perfis[df_perfis['Trilha'] != ""]
df_perfis = df_perfis[df_perfis['Skills'] != ""]


# 2. Definição do Perfil do Aluno (Simulação via Skills)
# Como a base PPC foi removida da arquitetura, o input do aluno é direto:
aluno_exemplo = {
    'nome': 'Estudante Teste ADS',
    'skills': ['Python', 'SQL', 'PostgreSQL', 'Docker', 'JavaScript'] 
}

skills_aluno_set = set(aluno_exemplo['skills'])
skills_aluno_text = ", ".join(aluno_exemplo['skills'])

print(f"==================================================")
print(f" ANÁLISE DE PERFIL: {aluno_exemplo['nome']}")
print(f"==================================================")
print(f"Skills mapeadas do aluno: {skills_aluno_text}\n")


# 3. Processamento de Linguagem Natural (NLP) e Vetorização
# Tokenizador customizado (mesmo de app.py)
def tokenizador_skills(texto):
    return [t.strip() for t in texto.split(",") if t.strip()]

documentos = list(df_perfis['Skills']) + [skills_aluno_text]

vectorizer = TfidfVectorizer(
    tokenizer=tokenizador_skills,
    lowercase=True,
    token_pattern=None
)

matriz = vectorizer.fit_transform(documentos)
matriz_trilhas = matriz[:-1]
vetor_aluno = matriz[-1]

# 4. Cálculo de Similaridade e Cobertura
similaridades = cosine_similarity(vetor_aluno, matriz_trilhas)[0]

resultados = []
for indice, row in df_perfis.iterrows():
    trilha = row['Trilha']
    skills_trilha = {skill.strip() for skill in str(row['Skills']).split(",") if skill.strip()}
    
    similaridade = float(similaridades[indice])
    
    # Cobertura: % das skills exigidas que o aluno possui
    cobertura = 0.0
    if skills_trilha:
        cobertura = len(skills_aluno_set.intersection(skills_trilha)) / len(skills_trilha)
        
    # Pontuação Final: 40% TF-IDF Cosine + 60% Cobertura
    pontuacao = (similaridade * 0.40) + (cobertura * 0.60)
    
    resultados.append({
        'Trilha': trilha,
        'Similaridade_%': similaridade * 100,
        'Cobertura_%': cobertura * 100,
        'Aderência_%': pontuacao * 100,
        'Skills_Trilha': skills_trilha
    })

df_resultados = pd.DataFrame(resultados).sort_values(by="Aderência_%", ascending=False)

print("--- Resultados de Aderência Global por Área de Mercado ---")
for _, row in df_resultados.iterrows():
    print(f"-> {row['Trilha']}: {row['Aderência_%']:.2f}% (Sim: {row['Similaridade_%']:.2f}%, Cob: {row['Cobertura_%']:.2f}%)")


# 5. Análise de Gaps 
print("\n--- Análise de Lacunas (Skills Gaps Recomendados) ---")
for _, row in df_resultados.iterrows():
    trilha = row['Trilha']
    skills_trilha = row['Skills_Trilha']
    
    gaps = sorted(list(skills_trilha - skills_aluno_set))
    print(f"[{trilha}] Competências a desenvolver: {gaps}")


# 6. Geração de Gráfico Analítico (Exportação para o Relatório)
plt.figure(figsize=(9, 5))
plt.barh(df_resultados['Trilha'], df_resultados['Aderência_%'], color='#2b5c8f')
plt.xlabel('Percentual de Aderência (%)', fontsize=12)
plt.title(f"Aderência Profissional - {aluno_exemplo['nome']}", fontsize=14, fontweight='bold')
plt.xlim(0, 100)
plt.gca().invert_yaxis() # Deixa a maior porcentagem no topo
plt.tight_layout()

# Salva o gráfico na pasta docs para uso no relatório do TDE
os.makedirs('../docs', exist_ok=True)
caminho_grafico = '../docs/grafico_aderencia.png'
plt.savefig(caminho_grafico, dpi=300)
print(f"\n[Sucesso] Gráfico analítico gerado e salvo em: {caminho_grafico}")