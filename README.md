# O Desafio do Classificador Mestre

Projeto de Aprendizagem de Máquina do curso de Desenvolvimento de Software Multiplataforma da FATEC Indaiatuba. O Grupo 2 compara modelos para classificar estudantes em evasão, permanência ou conclusão do curso.

**Integrantes:** Milton Wagner Filho, Gustavo Molina, João Emanoel e Murilo.

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

## Como abrir o dashboard no Windows

Com Git e Python 3.12 disponíveis, execute no PowerShell:

```powershell
git clone https://github.com/miltonwagner/4.2-ML-Desafio_Classificador_Mestre.git
Set-Location .\4.2-ML-Desafio_Classificador_Mestre
py -3.12 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit plotly jinja2
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
```

Abra o endereço **Local URL** exibido pelo Streamlit. O painel lê as métricas e figuras versionadas em `resultados/`. A aba **Escolha do modelo** mostra os tempos medidos durante a validação e simulações de crescimento da base, sob hipóteses linear e quadrática.

## Estrutura

| Caminho | Finalidade |
| --- | --- |
| `app.py` | Painel Streamlit com métricas, matrizes, figuras comentadas e metodologia |
| `analise_dados.py` | Análise exploratória, treinamento, ajuste e avaliação |
| `dataset.csv` | Base usada no projeto |
| `resultados/` | Tabelas, figuras e registro da escolha anterior ao teste |
| `modelos/` | Pipelines e metadados gerados localmente pelo Bloco 28 |

Para reproduzir o treinamento e gerar os pipelines, execute `& ".\.venv\Scripts\python.exe" analise_dados.py`. Feche as janelas dos gráficos à medida que aparecerem para que a execução prossiga.

## Referências

- [Dataset no UCI Machine Learning Repository](https://doi.org/10.24432/C5MC89)
- [Artigo que descreve a base](https://doi.org/10.3390/data7110146)
- [Referência para o período das matrículas](https://www.mdpi.com/2227-7390/14/4/626)
