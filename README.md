# Sistema Inteligente de Recomendação e Prescrição de Carreiras para ADS

Plataforma educacional baseada em **Inteligência Artificial** e **Processamento de Linguagem Natural (NLP)** desenvolvida como Trabalho de Discente Efetivo (TDE) no curso de Análise e Desenvolvimento de Sistemas do **UniFacema**.

O sistema analisa as competências técnicas declaradas pelo estudante, compara com dados reais do mercado de tecnologia e recomenda trilhas profissionais compatíveis com seu perfil, além de gerar um Roadmap prático com projetos escaláveis para preencher as lacunas identificadas.

---

## Trilhas Mapeadas

| # | Trilha |
|---|--------|
| 1 | Desenvolvimento Back-End e APIs |
| 2 | Desenvolvimento Front-End |
| 3 | Desenvolvimento Full-Stack |
| 4 | Engenharia de Dados e IA |
| 5 | Desenvolvimento Mobile |
| 6 | DevOps e Infraestrutura |
| 7 | Segurança da Informação (Cybersec) |
| 8 | Qualidade de Software (QA) |

---

## Arquitetura do Sistema

```
survey_results_public.csv
        │
        ▼
  etl_survey.py          ← ETL: filtra, pondera e extrai Top 20 skills por trilha
        │
        ▼
 perfis_mercado.CSV       ← Base consolidada de perfis de mercado
        │
        ▼
     app.py               ← Motor de recomendação + Interface Streamlit
        │
        ├── TF-IDF (40%)  ← Similaridade estatística de competências
        └── Cobertura (60%) ← Proporção real de skills adquiridas vs. exigidas
```

### Fórmula de Aderência

```
Aderência = (0,40 × Similaridade TF-IDF/Cosseno) + (0,60 × Cobertura)
```

---

## Fonte de Dados

- **[Stack Overflow Developer Survey](https://insights.stackoverflow.com/survey/)** — única fonte de dados do sistema.  
  O arquivo `survey_results_public.csv` deve ser baixado manualmente e colocado na pasta `data/`.

---

## Estrutura do Projeto

```
TdeNayron/
├── data/
│   ├── survey_results_public.csv   ← baixar manualmente (não versionado)
│   └── perfis_mercado.CSV          ← gerado pelo ETL
├── docs/
│   └── grafico_aderencia.png       ← gerado pelo main.py
├── src/
│   ├── etl_survey.py               ← pipeline de ETL
│   ├── app.py                      ← aplicação Streamlit
│   └── main.py                     ← simulador de perfil via terminal
├── requirements.txt
└── README.md
```

---

## Como Executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Gerar a base de mercado (ETL)

> Necessário apenas na primeira execução ou ao atualizar a survey.

```bash
cd src
python etl_survey.py
```

### 3. Iniciar a aplicação

```bash
cd src
streamlit run app.py
```

### (Opcional) Simulador via terminal

```bash
cd src
python main.py
```

---

## Equipe

| Nome | Contribuição |
|------|--------------|
| **Sara Victória Da Costa Silva** | Arquitetura de software, modelagem do algoritmo de IA (TF-IDF, Custom Tokenizer, métrica híbrida) e lógica prescritiva do Roadmap |
| **Alexandre Kauê Lima Amorim** | Revisão bibliográfica, fundamentação teórica, redação e formatação ABNT do relatório |
| **Jó Brandão Pimentel** | Pipeline de ETL, limpeza e tratamento da base da Stack Overflow com Pandas |
| **Phellipe Duarte Araújo** | Desenvolvimento do Front-End em Streamlit, UX/UI, testes de interface e roteiro do videocast |

---

## Tecnologias Utilizadas

- **Python 3.x**
- **Pandas** — ETL e manipulação de dados
- **Scikit-learn** — TF-IDF e Similaridade de Cosseno
- **Streamlit** — Interface web interativa
- **Matplotlib** — Geração de gráficos analíticos

---

## Referências

- MCKINNEY, Wes. *Python para análise de dados*. São Paulo: Novatec, 2018.
- SCIKIT-LEARN. *Machine Learning in Python*. Disponível em: https://scikit-learn.org/
- STACK OVERFLOW. *Developer Survey Results 2026*. Disponível em: https://insights.stackoverflow.com/survey/
- STREAMLIT. *The fastest way to build and share data apps*. Disponível em: https://streamlit.io/
