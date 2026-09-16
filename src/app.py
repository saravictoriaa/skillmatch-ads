from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Sistema Inteligente de Recomendação de Carreiras",
    page_icon="🎯",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
CAMINHO_PERFIS = BASE_DIR / "data" / "perfis_mercado.CSV"

MAX_ROADMAP = 8

ROADMAP_POR_TRILHA = {
    "Desenvolvimento Back-End e APIs": [
        ("PostgreSQL", "Otimizar consultas com índices B-Tree/GIN e particionamento de tabelas de grande volume."),
        ("Docker", "Containerizar uma API REST com Docker Compose integrando banco de dados e cache Redis."),
        ("Kubernetes", "Orquestrar uma API em cluster com Pods replicados, Ingress Controller e HPA."),
        ("Redis", "Implementar camada de cache distribuído e fila de mensagens (Pub/Sub) em uma API Python."),
        ("Terraform", "Provisionar infraestrutura de API em AWS (EC2 + RDS + Load Balancer) via IaC."),
        ("TypeScript", "Refatorar uma API Node.js com tipagem estrita, DTOs e validação de payloads."),
        ("Java", "Desenvolver API corporativa em Spring Boot com autenticação JWT e mensageria RabbitMQ."),
        ("Bash/Shell (all shells)", "Criar pipeline de CI/CD em shell script realizando build, testes e deploy em servidor remoto."),
    ],
    "Desenvolvimento Front-End": [
        ("TypeScript", "Migrar uma SPA JavaScript para TypeScript com tipos estritos e interfaces de componentes."),
        ("Vite", "Configurar projeto frontend com Vite + HMR e empacotamento otimizado via Rollup."),
        ("Tailwind CSS 4", "Desenvolver um Design System completo com Tailwind usando tokens de design estritos."),
        ("Webpack", "Configurar divisão de código (code splitting) e otimização de assets para produção."),
        ("Vercel", "Automatizar pipeline de CI/CD com preview deployments e Serverless Functions."),
        ("Shadcn/ui", "Criar biblioteca de componentes acessíveis e customizáveis com Shadcn + Radix UI."),
        ("Amazon Web Services (AWS)", "Hospedar e distribuir SPA utilizando S3 + CloudFront com invalidação de cache automática."),
        ("Bash/Shell (all shells)", "Automatizar build, lint e deploy de aplicações estáticas com scripts shell."),
    ],
    "Desenvolvimento Full-Stack": [
        ("TypeScript", "Desenvolver aplicação Full-Stack com tipagem compartilhada entre frontend e backend (monorepo)."),
        ("C#", "Construir API corporativa em ASP.NET Core com Entity Framework e autenticação OAuth2."),
        ("Microsoft SQL Server", "Modelar banco relacional corporativo com procedures, triggers e jobs automatizados."),
        ("Vite", "Integrar frontend moderno em Vite com backend Node.js usando proxy reverso configurado."),
        ("Microsoft Azure", "Realizar deploy Full-Stack no Azure com App Services, Azure SQL e CI/CD via GitHub Actions."),
        ("Tailwind CSS 4", "Desenvolver painel administrativo (dashboard) responsivo com Tailwind e componentes reutilizáveis."),
        ("Webpack", "Configurar micro-frontends com Module Federation para times de desenvolvimento escaláveis."),
        ("Yarn", "Gerenciar mono-repositório com múltiplos pacotes (frontend, backend, shared) via Yarn Workspaces."),
    ],
    "Engenharia de Dados e IA": [
        ("RAG", "Construir sistema de busca semântica combinando embeddings, banco vetorial (Pinecone/Weaviate) e LLM."),
        ("Large Language Model", "Realizar fine-tuning de modelo open-source (Llama/Mistral) com LoRA para domínio específico."),
        ("openAI GPT (chatbot models)", "Desenvolver agente de atendimento com memória persistente usando API do GPT e LangChain."),
        ("openAI Reasoning models", "Implementar fluxo de tomada de decisão complexo com modelos o1 da OpenAI para análise técnica."),
        ("Pydantic", "Validar e serializar dados de pipelines de ML garantindo integridade de schemas entre etapas."),
        ("Google Cloud", "Criar pipeline de dados serverless no GCP com Cloud Functions, Pub/Sub e BigQuery."),
        ("Microsoft Azure", "Arquitetar solução de ML na Azure com Azure ML, Data Factory e armazenamento no ADLS."),
        ("Gemini (Flash general purpose models)", "Construir pipeline de classificação de documentos em lote usando a API Flash do Gemini."),
    ],
    "Desenvolvimento Mobile": [
        ("Kotlin", "Desenvolver app Android com arquitetura MVVM, Retrofit para consumo de APIs e Coroutines."),
        ("Swift", "Criar aplicativo iOS nativo integrando SwiftUI com Combine para gerenciamento de estado reativo."),
        ("Dart", "Desenvolver app multiplataforma em Flutter com gerenciamento de estado via BLoC ou Riverpod."),
        ("Firebase", "Implementar autenticação OAuth2, banco NoSQL (Firestore) e Push Notifications no aplicativo."),
        ("Gradle", "Configurar builds multi-módulo com variantes de produto e integração de dependências em projetos Android."),
        ("Google Cloud", "Integrar serviços do GCP (Vision AI, Speech-to-Text) como funcionalidades nativas no aplicativo."),
        ("TypeScript", "Desenvolver aplicativo React Native com TypeScript, navegação tipada e gerenciamento de estado."),
        ("Amazon Web Services (AWS)", "Configurar autenticação (Cognito), armazenamento (S3) e notificações push (SNS) para o App."),
    ],
    "DevOps e Infraestrutura": [
        ("Terraform", "Provisionar ambiente Cloud completo (VPC, subnets, EC2, RDS) com state remoto no S3."),
        ("Ansible", "Criar playbooks para hardening e configuração padronizada de múltiplos servidores Linux."),
        ("Prometheus", "Configurar monitoramento de métricas de microserviços com alertas via Alertmanager."),
        ("Go", "Desenvolver ferramenta CLI de alta performance para automação de tarefas de infraestrutura."),
        ("Make", "Estruturar Makefile profissional para orquestrar builds, testes, lint e deploys com um único comando."),
        ("APT", "Gerenciar repositórios privados e automação de pacotes em distribuições Debian para ambientes corporativos."),
        ("Microsoft Azure", "Implementar infraestrutura como código na Azure com Bicep e pipelines de CI/CD no Azure DevOps."),
        ("Bash/Shell (all shells)", "Desenvolver rotinas de backup automatizado, rotação de logs e monitoramento de processos críticos."),
    ],
    "Segurança da Informação (Cybersec)": [
        ("Python", "Desenvolver scanner de vulnerabilidades automatizado com análise de portas e fingerprinting de serviços."),
        ("Cloudflare", "Configurar regras de WAF, proteção contra DDoS e políticas de acesso baseadas em geolocalização."),
        ("PowerShell", "Criar scripts de auditoria e endurecimento (hardening) de ambientes Windows Server em escala."),
        ("C", "Desenvolver exploit de prova de conceito (PoC) para análise de vulnerabilidades de baixo nível."),
        ("APT", "Gerenciar repositórios seguros e automatizar aplicação de patches de segurança em servidores Debian."),
        ("Kubernetes", "Implementar políticas de segurança de cluster (NetworkPolicy, PodSecurityPolicy e RBAC)."),
        ("Microsoft Azure", "Configurar Azure Sentinel para SIEM, detecção de ameaças e automação de resposta a incidentes."),
        ("Java", "Desenvolver aplicação segura em Spring Security com proteção contra OWASP Top 10 e auditoria de acessos."),
    ],
    "Qualidade de Software (QA)": [
        ("Python", "Criar suíte completa de testes E2E com Playwright e relatórios automatizados de cobertura."),
        ("Java", "Desenvolver framework de testes de performance e carga com JMeter integrado ao pipeline de CI."),
        ("Maven (build tool)", "Configurar ciclo de vida de testes com Maven, Surefire e geração de relatórios Allure."),
        ("TypeScript", "Implementar testes E2E tipados com Cypress e fixtures reutilizáveis para aplicações web complexas."),
        ("Microsoft SQL Server", "Criar massa de dados automatizada e validar consistência de dados após migrações de banco."),
        ("PowerShell", "Automatizar execução de testes de regressão e geração de relatórios em ambientes Windows."),
        ("Amazon Web Services (AWS)", "Executar testes de carga em ambiente de homologação na nuvem usando AWS Device Farm ou Locust."),
        ("Bash/Shell (all shells)", "Criar hooks de pre-commit para validação de lint, formatação e testes unitários antes do push."),
    ],
}


@st.cache_data
def carregar_perfis():
    if not CAMINHO_PERFIS.exists():
        st.error(f"Arquivo não encontrado: `{CAMINHO_PERFIS}`. Execute `etl_survey.py` primeiro.")
        st.stop()
    try:
        df = pd.read_csv(CAMINHO_PERFIS, sep=";", encoding="utf-8")
    except Exception as erro:
        st.error(f"Erro ao carregar a base: {erro}")
        st.stop()

    if not {"Trilha", "Skills"}.issubset(df.columns):
        st.error("Colunas `Trilha` e `Skills` não encontradas em `perfis_mercado.CSV`.")
        st.stop()

    df = df.copy()
    df["Trilha"] = df["Trilha"].astype(str).str.strip()
    df["Skills"] = df["Skills"].astype(str).str.strip()
    df = df[(df["Trilha"] != "") & (df["Skills"] != "")]
    return df


def extrair_vocabulario(df):
    skills = set()
    for valor in df["Skills"]:
        for skill in str(valor).split(","):
            skill = skill.strip()
            if skill:
                skills.add(skill)
    return sorted(skills, key=str.lower)


def separar_skills(texto):
    if pd.isna(texto):
        return set()
    return {skill.strip() for skill in str(texto).split(",") if skill.strip()}


def calcular_cobertura(skills_aluno, skills_trilha):
    if not skills_trilha:
        return 0.0
    return len(skills_aluno.intersection(skills_trilha)) / len(skills_trilha)


def calcular_similaridade_tfidf(perfis, texto_aluno):
    documentos = list(perfis["Skills"]) + [texto_aluno]

    def tokenizador_skills(texto):
        return [t.strip() for t in texto.split(",") if t.strip()]

    vectorizer = TfidfVectorizer(tokenizer=tokenizador_skills, lowercase=True, token_pattern=None)
    matriz = vectorizer.fit_transform(documentos)
    return cosine_similarity(matriz[-1], matriz[:-1])[0]


def gerar_roadmap(trilha, gaps):
    sugestoes_trilha = {skill: projeto for skill, projeto in ROADMAP_POR_TRILHA.get(trilha, [])}
    roadmap = []
    for gap in gaps:
        if gap in sugestoes_trilha:
            roadmap.append((gap, sugestoes_trilha[gap]))
        if len(roadmap) == MAX_ROADMAP:
            break
    return roadmap


df_perfis = carregar_perfis()
vocabulario_skills = extrair_vocabulario(df_perfis)
trilhas_disponiveis = df_perfis["Trilha"].tolist()

st.title("🎯 Sistema Inteligente de Recomendação de Carreiras para ADS")
st.markdown("Análise de compatibilidade com trilhas profissionais baseada nos dados do Stack Overflow Developer Survey.")
st.divider()

col_perfil, col_interesse = st.columns(2)

with col_perfil:
    st.header("1. Perfil técnico")
    skills_aluno = st.multiselect(
        "Selecione suas competências:",
        options=vocabulario_skills,
        help="Competências extraídas da base de mercado."
    )

with col_interesse:
    st.header("2. Áreas de interesse")
    areas_interesse = st.multiselect(
        "Selecione até 3 áreas (opcional):",
        options=trilhas_disponiveis,
        max_selections=3,
        help="Contexto declarado — não altera a pontuação estatística."
    )

st.divider()

executar_analise = st.button("🚀 Analisar perfil e gerar Roadmap", type="primary", use_container_width=True)

if executar_analise:
    if not skills_aluno:
        st.warning("Selecione pelo menos uma competência técnica para realizar a análise.")
        st.stop()

    texto_aluno = ", ".join(skills_aluno)
    similaridades = calcular_similaridade_tfidf(df_perfis, texto_aluno)

    resultados = []
    for indice, (_, linha) in enumerate(df_perfis.iterrows()):
        trilha = linha["Trilha"]
        skills_trilha = separar_skills(linha["Skills"])
        similaridade = float(similaridades[indice])
        cobertura = calcular_cobertura(set(skills_aluno), skills_trilha)
        pontuacao = (similaridade * 0.40) + (cobertura * 0.60)
        resultados.append({
            "Trilha": trilha,
            "Similaridade": similaridade,
            "Cobertura": cobertura,
            "Pontuação": pontuacao,
            "Skills": skills_trilha,
        })

    df_resultados = pd.DataFrame(resultados).sort_values(by="Pontuação", ascending=False).reset_index(drop=True)

    st.header("3. Resultado da recomendação")
    for _, resultado in df_resultados.iterrows():
        trilha = resultado["Trilha"]
        pontuacao = resultado["Pontuação"]
        cobertura = resultado["Cobertura"]
        similaridade = resultado["Similaridade"]

        with st.container(border=True):
            col1, col2, col3 = st.columns([2.5, 1, 1])
            with col1:
                st.subheader(f"⭐ {trilha}" if trilha in areas_interesse else trilha)
            with col2:
                st.metric("Aderência Global", f"{pontuacao * 100:.2f}%")
            with col3:
                st.metric("Cobertura Específica", f"{cobertura * 100:.2f}%")
            st.progress(min(max(float(pontuacao), 0.0), 1.0))

    st.divider()
    st.header("4. Plano de Ação e Roadmap Profissional")

    for _, resultado in df_resultados.iterrows():
        trilha = resultado["Trilha"]
        skills_trilha = resultado["Skills"]
        gaps = sorted(skills_trilha - set(skills_aluno), key=str.lower)
        roadmap = gerar_roadmap(trilha, gaps)

        with st.expander(f"💼 {trilha}"):
            if not roadmap:
                st.success("Cobertura completa nas competências prioritárias desta trilha.")
                continue
            for skill, projeto in roadmap:
                st.markdown(f"**{skill}**: {projeto}")

    st.divider()
    with st.expander("🔎 Detalhamento técnico"):
        st.markdown(
            "**Fórmula:** `Aderência = (0,40 × TF-IDF/Cosseno) + (0,60 × Cobertura)`\n\n"
            "**TF-IDF (40%):** relevância estatística das skills em comum.\n\n"
            "**Cobertura (60%):** `skills do aluno ∩ trilha / total da trilha`."
        )
        tabela = df_resultados[["Trilha", "Similaridade", "Cobertura", "Pontuação"]].copy()
        tabela["Similaridade"] = (tabela["Similaridade"] * 100).round(2)
        tabela["Cobertura"] = (tabela["Cobertura"] * 100).round(2)
        tabela["Pontuação"] = (tabela["Pontuação"] * 100).round(2)
        tabela = tabela.rename(columns={"Similaridade": "TF-IDF (%)", "Cobertura": "Cobertura (%)", "Pontuação": "Aderência (%)"})
        st.dataframe(tabela, use_container_width=True, hide_index=True)

st.divider()
st.caption("Sistema Inteligente de Recomendação e Prescrição de Carreiras para ADS")