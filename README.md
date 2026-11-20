# O Desafio do Classificador Mestre

Projeto de Aprendizagem de Máquina do curso de Desenvolvimento de Software Multiplataforma da FATEC Indaiatuba. O Grupo 2 compara modelos para classificar estudantes em evasão, permanência ou conclusão do curso.

**Integrantes:** Milton Wagner Filho, Gustavo Molina, João Emanoel e Murilo.

## Sumário

- [Dashboard Streamlit](#dashboard-streamlit)
- [Dados](#dados)
- [Metodologia e resultado](#metodologia-e-resultado)
- [Como executar no PyCharm](#como-executar-no-pycharm)
- [Como reproduzir a análise](#como-reproduzir-a-análise)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Referências](#referências)

## Dashboard Streamlit

**[Abrir o dashboard público](https://classificador-mestre-fatec.streamlit.app)** — acesso pelo navegador, sem login.

| Aba | Conteúdo |
| --- | --- |
| Comparação de modelos | Métricas dos sete classificadores no conjunto de teste |
| Acertos e erros | Matriz de confusão e interpretação dos erros da Regressão Logística |
| Escolha do modelo | Critério de seleção, tempos da validação cruzada e simulações para bases maiores |
| Figuras do projeto | Gráficos comentados, incluindo as matrizes do teste (Figura 7) e o desempenho acadêmico por classe no treino (Figura 8) |
| Metodologia | Origem dos dados, divisão de treino e teste, avaliação e pré-processamento |

O dashboard lê os CSVs e as figuras em `resultados/`. Para abrir o painel, basta executar o Streamlit com `app.py`.

[Consultar o relatório em PDF](docs/Relatorio_P2_Classificador_Mestre.pdf)

## Dados

A base reúne 4.424 registros de estudantes do Instituto Politécnico de Portalegre, no Alto Alentejo, Portugal, com matrículas entre os anos letivos de 2008/2009 e 2018/2019. O arquivo `dataset.csv` contém 4.424 linhas de dados, além do cabeçalho, e 35 colunas: 34 variáveis preditoras e a variável alvo `Target`. O artigo que descreve a base foi publicado em 2022. A auditoria encontrou zero valores ausentes e zero linhas integralmente duplicadas.

| Classe | Significado | Registros |
| --- | --- | ---: |
| `Dropout` | Evasão | 1.421 |
| `Enrolled` | Permaneceu matriculado | 794 |
| `Graduate` | Concluiu o curso | 2.209 |

As entradas incluem o desempenho nos dois primeiros semestres. Os resultados devem ser interpretados no contexto dessa instituição e desse período.

## Metodologia e resultado

- Divisão aleatória estratificada de 70% para treino (3.096) e 30% para teste (1.328), com `random_state=42`.
- Pré-processamento com padronização de variáveis numéricas, codificação one-hot das categóricas e passagem direta das binárias.
- Ajuste de hiperparâmetros com validação cruzada estratificada em cinco divisões no treino. Foram avaliados KNN, Regressão Logística, Árvore de Decisão, Random Forest, SVM linear, SVM RBF e Gaussian Naive Bayes.
- A Regressão Logística foi escolhida antes do teste por obter o maior F1 macro médio na validação (**71,41%**). O F1 macro dá o mesmo peso a cada classe.
- No teste final, a Regressão Logística obteve **70,80%** de F1 macro. O Random Forest obteve o maior valor observado nessa etapa, **72,42%**.

## Como executar no PyCharm

**Pré-requisitos** — PyCharm, Git e Python 3.12.

### 1. Abrir o projeto

Se ainda não tiver o projeto, clone o repositório. O exemplo usa a pasta `Documents` do usuário.

```powershell
$projeto = Join-Path $env:USERPROFILE "Documents\4.2-ML-Desafio_Classificador_Mestre"
git clone https://github.com/miltonwagner/4.2-ML-Desafio_Classificador_Mestre.git "$projeto"
```

No PyCharm, selecione **File → Open** e abra a pasta do projeto. Em seguida, abra a aba **Terminal** com o PowerShell.

### 2. Preparar o ambiente

Defina `$projeto` com o caminho absoluto da pasta que você abriu. O exemplo abaixo usa a pasta de trabalho do projeto na FATEC.

```powershell
$projeto = "C:\Fatec\Aulas 5 Semestre\Aprendizado de Maquina\P2 Seminario Projeto\4.2-ML-Desafio_Classificador_Mestre"
Set-Location -LiteralPath "$projeto"
if (-not (Test-Path -LiteralPath "$projeto\.venv\Scripts\python.exe")) {
    py -3.12 -m venv "$projeto\.venv"
}
& "$projeto\.venv\Scripts\python.exe" -m pip install -r "$projeto\requirements.txt"
```

### 3. Iniciar o dashboard

No mesmo terminal, execute:

```powershell
& "$projeto\.venv\Scripts\python.exe" -m streamlit run "$projeto\app.py"
```

Abra o endereço **Local URL** exibido no terminal, normalmente `http://localhost:8501`. Use `Ctrl+C` no terminal para encerrar o dashboard.

## Como reproduzir a análise

Para executar a análise e o treinamento, instale também as dependências abaixo no ambiente preparado na seção anterior.

```powershell
& "$projeto\.venv\Scripts\python.exe" -m pip install numpy scikit-learn matplotlib seaborn joblib
& "$projeto\.venv\Scripts\python.exe" "$projeto\analise_dados.py"
```

Feche as janelas dos gráficos à medida que aparecerem para que a execução prossiga. O script regenera arquivos em `resultados/` e o Bloco 28 exporta os pipelines e metadados em `modelos/`.

## Estrutura do repositório

| Caminho | Finalidade |
| --- | --- |
| `README.md` | Visão geral e instruções de execução |
| `requirements.txt` | Dependências do dashboard |
| `Dockerfile` e `.dockerignore` | Configuração para execução local em contêiner |
| `app.py` | Painel Streamlit com métricas, matrizes, figuras comentadas e metodologia |
| `analise_dados.py` | Análise exploratória, treinamento, ajuste e avaliação |
| `dataset.csv` | Base usada no projeto |
| `resultados/` | Tabelas, figuras e registro da escolha anterior ao teste |
| `modelos/` | Pipelines e metadados gerados localmente pelo Bloco 28 |

## Referências

- [Dataset no UCI Machine Learning Repository](https://doi.org/10.24432/C5MC89)
- [Artigo que descreve a base](https://doi.org/10.3390/data7110146)
- [Referência para o período das matrículas](https://www.mdpi.com/2227-7390/14/4/626)
