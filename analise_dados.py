"""
FATEC — Aprendizagem de Máquina
Projeto: O Desafio do Classificador Mestre
Grupo 2: Milton, Gustavo Molina, João Emanoel e Murilo

Tema: Predição de evasão e sucesso acadêmico.

Execute os blocos na ordem: os seguintes dependem dos anteriores.
"""

# %% ==========================================================
# BLOCO 1 — Bibliotecas e configurações
# =============================================================

import sys
from pathlib import Path

import pandas as pd


# O CSV deve ficar na mesma pasta deste arquivo Python.
PASTA_PROJETO = Path(__file__).resolve().parent
ARQUIVO_DADOS = PASTA_PROJETO / "dataset.csv"

ALVO = "Target"
CLASSES_ESPERADAS = {"Dropout", "Enrolled", "Graduate"}

SIGNIFICADOS = {
    "Dropout": "Abandonou o curso",
    "Enrolled": "Permanece matriculado",
    "Graduate": "Formado",
}

print("=" * 70)
print("BLOCO 1 — BIBLIOTECAS E CONFIGURAÇÕES")
print("=" * 70)

print("Python utilizado:", sys.executable)
print("Versão do Python:", sys.version.split()[0])
print("Versão do Pandas:", pd.__version__)
print("Pasta do projeto:", PASTA_PROJETO)


# %% ==========================================================
# BLOCO 2 — Leitura e auditoria dos dados
# =============================================================

print("\n" + "=" * 70)
print("BLOCO 2 — LEITURA E AUDITORIA DOS DADOS")
print("=" * 70)

if not ARQUIVO_DADOS.is_file():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {ARQUIVO_DADOS}"
    )

# Mantém os dados originais, sem limpeza ou remoção de registros.
bruto = pd.read_csv(ARQUIVO_DADOS)

if ALVO not in bruto.columns:
    raise ValueError(
        f"O dataset precisa conter a coluna '{ALVO}'."
    )

if bruto.empty:
    raise ValueError("O dataset não contém registros.")

print("\nArquivo:", ARQUIVO_DADOS.name)
print("Quantidade de registros:", bruto.shape[0])
print("Quantidade de colunas:", bruto.shape[1])
print("Quantidade de variáveis de entrada:", bruto.shape[1] - 1)
print("Variável alvo:", ALVO)

print("\nPrimeiros cinco registros:")
print(bruto.head().to_string(index=False))

nulos_por_coluna = bruto.isna().sum()
colunas_com_nulos = nulos_por_coluna[nulos_por_coluna > 0]

print("\nTotal de valores ausentes:", int(nulos_por_coluna.sum()))
print("Linhas inteiramente duplicadas:", int(bruto.duplicated().sum()))

if colunas_com_nulos.empty:
    print("Nenhuma coluna apresentou valores ausentes.")
else:
    print("\nValores ausentes por coluna:")
    print(colunas_com_nulos.to_string())

print("\nTipos das colunas:")
print(bruto.dtypes.to_string())

print(
    "\nObservação: colunas numéricas podem representar códigos de "
    "categorias, como curso e estado civil."
)


# %% ==========================================================
# BLOCO 3 — Distribuição e validação das classes
# =============================================================

print("\n" + "=" * 70)
print("BLOCO 3 — DISTRIBUIÇÃO DAS CLASSES")
print("=" * 70)

contagens = bruto[ALVO].value_counts(dropna=False)

distribuicao = pd.DataFrame({
    "Quantidade": contagens,
    "Percentual (%)": (contagens / len(bruto) * 100).round(2),
})

distribuicao.index.name = ALVO
distribuicao["Significado"] = [
    SIGNIFICADOS.get(classe, "Categoria não reconhecida")
    for classe in distribuicao.index
]

print("\n" + distribuicao.to_string())

classes_encontradas = set(bruto[ALVO].dropna().unique())

if classes_encontradas != CLASSES_ESPERADAS:
    raise ValueError(
        "As classes encontradas diferem das esperadas.\n"
        f"Esperadas: {sorted(CLASSES_ESPERADAS)}\n"
        f"Encontradas: {sorted(classes_encontradas)}"
    )

if bruto[ALVO].isna().any():
    raise ValueError("Existem registros sem a categoria alvo.")

print("\nAs três classes esperadas foram confirmadas.")
print(
    "\nCompare as quantidades acima para avaliar o equilíbrio "
    "entre as classes."
)

print("\nBlocos 1 a 3 concluídos. Nenhum dado foi alterado.")

# %% ==========================================================
# BLOCO 4 — Gráfico da distribuição das classes
# =============================================================

import matplotlib.pyplot as plt

print("\n" + "=" * 70)
print("BLOCO 4 — GRÁFICO DA DISTRIBUIÇÃO DAS CLASSES")
print("=" * 70)

# Cria a pasta de resultados, caso ainda não exista.
PASTA_RESULTADOS = PASTA_PROJETO / "resultados"
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

# Define uma ordem fixa para as categorias.
ordem_classes = ["Dropout", "Enrolled", "Graduate"]

quantidades = bruto[ALVO].value_counts().reindex(ordem_classes)
percentuais = quantidades / len(bruto) * 100

rotulos = [
    "Abandonou\n(Dropout)",
    "Matriculado\n(Enrolled)",
    "Formado\n(Graduate)",
]

fig, ax = plt.subplots(figsize=(9, 5.5))

barras = ax.bar(
    rotulos,
    quantidades.to_numpy(),
    color=["#C44E52", "#E5A629", "#3A8D70"],
    width=0.6,
)

# Mostra a quantidade e o percentual acima de cada barra.
for barra, quantidade, percentual in zip(
    barras, quantidades, percentuais
):
    ax.annotate(
        f"{int(quantidade):,}".replace(",", ".")
        + f"\n({percentual:.2f}%)".replace(".", ","),
        xy=(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height(),
        ),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
    )

ax.set_title(
    "Distribuição da situação acadêmica dos estudantes",
    fontsize=14,
    pad=16,
)
ax.set_xlabel("Situação acadêmica")
ax.set_ylabel("Quantidade de estudantes")
ax.set_ylim(0, quantidades.max() * 1.2)
ax.yaxis.grid(True, alpha=0.25)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.text(
    0.5,
    0.02,
    "Fonte: dataset.csv — elaboração do grupo. Total: "
    f"{len(bruto):,} estudantes.".replace(",", "."),
    ha="center",
    fontsize=9,
)

fig.tight_layout(rect=(0, 0.06, 1, 1))

# Salva antes de abrir a janela do gráfico.
arquivo_grafico = (
    PASTA_RESULTADOS / "figura_01_distribuicao_classes.png"
)
fig.savefig(arquivo_grafico, dpi=300, bbox_inches="tight")

# Salva também os números usados na figura.
arquivo_tabela = PASTA_RESULTADOS / "distribuicao_classes.csv"
distribuicao.to_csv(
    arquivo_tabela,
    sep=";",
    encoding="utf-8-sig",
    decimal=",",
)

print("\nGráfico salvo em:", arquivo_grafico)
print("Tabela salva em:", arquivo_tabela)
print("\nFeche a janela do gráfico para concluir a execução.")

plt.show()
plt.close(fig)

# %% ==========================================================
# BLOCO 5 — Estatísticas descritivas das variáveis quantitativas
# =============================================================

print("\n" + "=" * 70)
print("BLOCO 5 — ESTATÍSTICAS DESCRITIVAS")
print("=" * 70)

# Seleção explícita: códigos de categorias não são medidas.
VARIAVEIS_QUANTITATIVAS = [
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# Confere os nomes antes de calcular as estatísticas.
colunas_faltantes = [
    coluna
    for coluna in VARIAVEIS_QUANTITATIVAS
    if coluna not in bruto.columns
]

if colunas_faltantes:
    raise ValueError(
        "Variáveis quantitativas não encontradas no dataset: "
        + ", ".join(colunas_faltantes)
    )

# Impede que valores textuais sejam ignorados silenciosamente.
colunas_nao_numericas = [
    coluna
    for coluna in VARIAVEIS_QUANTITATIVAS
    if not pd.api.types.is_numeric_dtype(bruto[coluna])
]

if colunas_nao_numericas:
    raise TypeError(
        "As seguintes variáveis precisam ser numéricas: "
        + ", ".join(colunas_nao_numericas)
    )

# .T coloca cada variável em uma linha da tabela.
estatisticas_quantitativas = (
    bruto[VARIAVEIS_QUANTITATIVAS]
    .describe(percentiles=[0.25, 0.50, 0.75])
    .T
    .rename(columns={
        "count": "Quantidade válida",
        "mean": "Média",
        "std": "Desvio padrão",
        "min": "Mínimo",
        "25%": "1º quartil (25%)",
        "50%": "Mediana (50%)",
        "75%": "3º quartil (75%)",
        "max": "Máximo",
    })
)

estatisticas_quantitativas.index.name = "Variável"
estatisticas_quantitativas["Quantidade válida"] = (
    estatisticas_quantitativas["Quantidade válida"].astype(int)
)

print(
    "\nQuantidade de variáveis quantitativas analisadas:",
    len(VARIAVEIS_QUANTITATIVAS),
)

# Arredondamento apenas para exibição no terminal.
print("\n" + estatisticas_quantitativas.round(2).to_string())

arquivo_estatisticas = (
    PASTA_RESULTADOS / "estatisticas_descritivas_quantitativas.csv"
)

estatisticas_quantitativas.to_csv(
    arquivo_estatisticas,
    sep=";",
    encoding="utf-8-sig",
    decimal=",",
    float_format="%.4f",
)

print("\nTabela salva em:", arquivo_estatisticas)
print(
    "\nObservação: este resumo não remove valores extremos "
    "nem altera os dados originais."
)
print("\nBloco 5 concluído.")

# %% ==========================================================
# BLOCO 6 — Distribuições e valores potencialmente extremos
# =============================================================

print("\n" + "=" * 70)
print("BLOCO 6 — DISTRIBUIÇÕES E VALORES EXTREMOS")
print("=" * 70)

VARIAVEIS_DISTRIBUICAO = [
    (
        "Age at enrollment",
        "Idade na matrícula",
        "Idade (anos)",
    ),
    (
        "Curricular units 1st sem (grade)",
        "Nota média do 1º semestre",
        "Nota",
    ),
    (
        "Curricular units 2nd sem (grade)",
        "Nota média do 2º semestre",
        "Nota",
    ),
]

cores_distribuicao = ["#4472C4", "#3A8D70", "#E5A629"]
resumo_extremos = []

# O layout ajusta os espaços quando a janela é redimensionada.
# A última linha fica reservada para a fonte e a observação.
fig_distribuicoes = plt.figure(
    figsize=(12, 8),
    layout="constrained",
)

fig_distribuicoes.canvas.manager.set_window_title(
    "Figura 2 — Distribuições de idade e notas"
)

grade = fig_distribuicoes.add_gridspec(
    nrows=4,
    ncols=2,
    height_ratios=[1, 1, 1, 0.22],
)

fig_distribuicoes.suptitle(
    "Distribuições de idade e notas acadêmicas",
    fontsize=14,
)

for linha, (coluna, descricao, unidade) in enumerate(
    VARIAVEIS_DISTRIBUICAO
):
    # Seleciona valores presentes sem alterar o dataset original.
    serie = bruto[coluna].dropna()

    if serie.empty:
        raise ValueError(
            f"A variável '{coluna}' não possui valores válidos."
        )

    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    abaixo = int((serie < limite_inferior).sum())
    acima = int((serie > limite_superior).sum())
    total_extremos = abaixo + acima

    resumo_extremos.append({
        "Variável": coluna,
        "Descrição": descricao,
        "Quantidade válida": len(serie),
        "Quantidade de zeros": int((serie == 0).sum()),
        "1º quartil": q1,
        "3º quartil": q3,
        "IQR": iqr,
        "Limite inferior": limite_inferior,
        "Limite superior": limite_superior,
        "Abaixo do limite": abaixo,
        "Acima do limite": acima,
        "Total potencialmente extremo": total_extremos,
        "Percentual potencialmente extremo (%)": (
            total_extremos / len(serie) * 100
        ),
    })

    cor = cores_distribuicao[linha]

    # Histograma.
    ax_hist = fig_distribuicoes.add_subplot(grade[linha, 0])

    ax_hist.hist(
        serie,
        bins=25,
        color=cor,
        edgecolor="white",
        alpha=0.85,
    )

    ax_hist.axvline(
        serie.mean(),
        color="#B22222",
        linestyle="--",
        linewidth=1.5,
        label=f"Média: {serie.mean():.2f}",
    )

    ax_hist.axvline(
        serie.median(),
        color="#222222",
        linestyle=":",
        linewidth=1.5,
        label=f"Mediana: {serie.median():.2f}",
    )

    ax_hist.set_title(
        f"{descricao} — distribuição",
        fontsize=11,
        pad=8,
    )
    ax_hist.set_xlabel(unidade, fontsize=9)
    ax_hist.set_ylabel("Quantidade de estudantes", fontsize=9)
    ax_hist.tick_params(labelsize=9)
    ax_hist.legend(fontsize=8)
    ax_hist.grid(axis="y", alpha=0.25)
    ax_hist.set_axisbelow(True)

    # Boxplot com limites pela regra de 1,5 × IQR.
    ax_box = fig_distribuicoes.add_subplot(grade[linha, 1])

    ax_box.boxplot(
        serie.to_numpy(),
        whis=1.5,
        patch_artist=True,
        boxprops={
            "facecolor": cor,
            "alpha": 0.65,
        },
        medianprops={
            "color": "#222222",
            "linewidth": 2,
        },
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "#B22222",
            "markeredgecolor": "#B22222",
            "alpha": 0.35,
        },
    )

    ax_box.set_title(
        f"{descricao} — boxplot",
        fontsize=11,
        pad=8,
    )
    ax_box.set_ylabel(unidade, fontsize=9)
    ax_box.set_xticks([])
    ax_box.tick_params(labelsize=9)
    ax_box.grid(axis="y", alpha=0.25)
    ax_box.set_axisbelow(True)

# Rodapé em uma área própria, fora dos gráficos.
ax_rodape = fig_distribuicoes.add_subplot(grade[3, :])
ax_rodape.set_axis_off()

ax_rodape.text(
    0.5,
    0.5,
    "Fonte: dataset.csv — elaboração do grupo.\n"
    "Pontos além dos limites do boxplot não indicam "
    "necessariamente erros.",
    transform=ax_rodape.transAxes,
    ha="center",
    va="center",
    fontsize=9,
)

tabela_extremos = pd.DataFrame(resumo_extremos)

print("\nResumo pela regra de 1,5 × IQR:")
print(tabela_extremos.round(2).to_string(index=False))

arquivo_distribuicoes = (
    PASTA_RESULTADOS / "figura_02_distribuicoes_idade_notas.png"
)

fig_distribuicoes.savefig(
    arquivo_distribuicoes,
    dpi=300,
    bbox_inches="tight",
)

arquivo_extremos = (
    PASTA_RESULTADOS / "resumo_valores_extremos_idade_notas.csv"
)

tabela_extremos.to_csv(
    arquivo_extremos,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    decimal=",",
    float_format="%.4f",
)

print("\nFigura salva em:", arquivo_distribuicoes)
print("Tabela salva em:", arquivo_extremos)

print(
    "\nAtenção: zeros nas notas e valores extremos precisam "
    "de interpretação. Nenhum registro foi removido."
)

print("\nFeche a janela da Figura 2 para concluir o Bloco 6.")

plt.show()
plt.close(fig_distribuicoes)

print("\nBloco 6 concluído.")

# %% ==========================================================
# BLOCO 7 — Investigação das notas zero
# =============================================================

print("\n" + "=" * 70)
print("BLOCO 7 — INVESTIGAÇÃO DAS NOTAS ZERO")
print("=" * 70)

SEMESTRES_INVESTIGACAO = [
    ("1º semestre", "1st"),
    ("2º semestre", "2nd"),
]

resumo_notas_zero = []
tabelas_zero_por_classe = []
tabelas_contexto_academico = []

for descricao_semestre, identificador in SEMESTRES_INVESTIGACAO:
    prefixo = f"Curricular units {identificador} sem"

    coluna_nota = f"{prefixo} (grade)"
    coluna_aprovadas = f"{prefixo} (approved)"
    coluna_avaliacoes = f"{prefixo} (evaluations)"
    coluna_inscritas = f"{prefixo} (enrolled)"
    coluna_sem_avaliacoes = f"{prefixo} (without evaluations)"

    colunas_necessarias = [
        ALVO,
        coluna_nota,
        coluna_aprovadas,
        coluna_avaliacoes,
        coluna_inscritas,
        coluna_sem_avaliacoes,
    ]

    faltantes = [
        coluna
        for coluna in colunas_necessarias
        if coluna not in bruto.columns
    ]

    if faltantes:
        raise ValueError(
            "Colunas não encontradas: " + ", ".join(faltantes)
        )

    # Máscara identifica registros; não modifica o dataset.
    mascara_zero = bruto[coluna_nota].eq(0)
    quantidade_zero = int(mascara_zero.sum())

    print("\n" + "-" * 70)
    print(descricao_semestre.upper())
    print("-" * 70)

    print("Registros com nota zero:", quantidade_zero)
    print(
        "Percentual sobre todos os estudantes:",
        f"{quantidade_zero / len(bruto) * 100:.2f}%",
    )

    resumo_notas_zero.append({
        "Semestre": descricao_semestre,
        "Total de estudantes": len(bruto),
        "Notas válidas": int(bruto[coluna_nota].notna().sum()),
        "Notas ausentes": int(bruto[coluna_nota].isna().sum()),
        "Quantidade com nota zero": quantidade_zero,
        "Percentual com nota zero (%)": (
            quantidade_zero / len(bruto) * 100
        ),
        "Nota zero e aprovações iguais a zero": int(
            (
                mascara_zero
                & bruto[coluna_aprovadas].eq(0)
            ).sum()
        ),
        "Nota zero e aprovações maiores que zero": int(
            (
                mascara_zero
                & bruto[coluna_aprovadas].gt(0)
            ).sum()
        ),
        "Nota zero e avaliações iguais a zero": int(
            (
                mascara_zero
                & bruto[coluna_avaliacoes].eq(0)
            ).sum()
        ),
        "Nota zero e avaliações maiores que zero": int(
            (
                mascara_zero
                & bruto[coluna_avaliacoes].gt(0)
            ).sum()
        ),
        "Nota zero e inscrições iguais a zero": int(
            (
                mascara_zero
                & bruto[coluna_inscritas].eq(0)
            ).sum()
        ),
    })

    # Compara a presença de notas zero dentro de cada classe.
    totais_classe = (
        bruto[ALVO]
        .value_counts()
        .reindex(ordem_classes, fill_value=0)
    )

    zeros_classe = (
        bruto.loc[mascara_zero, ALVO]
        .value_counts()
        .reindex(ordem_classes, fill_value=0)
    )

    tabela_classe = pd.DataFrame({
        "Total da classe": totais_classe,
        "Quantidade com nota zero": zeros_classe,
    })

    # Evita divisão por zero caso alguma classe esteja vazia.
    denominador = tabela_classe["Total da classe"].replace(
        0, float("nan")
    )

    tabela_classe["Percentual com zero dentro da classe (%)"] = (
        tabela_classe["Quantidade com nota zero"]
        / denominador
        * 100
    )

    tabela_classe.index.name = ALVO
    tabela_classe = tabela_classe.reset_index()
    tabela_classe.insert(0, "Semestre", descricao_semestre)

    tabelas_zero_por_classe.append(tabela_classe)

    print("\nNotas zero por situação acadêmica:")
    print(tabela_classe.round(2).to_string(index=False))

    # Resume os dados acadêmicos somente dos registros com nota zero.
    colunas_contexto = {
        coluna_inscritas: "Unidades inscritas",
        coluna_avaliacoes: "Avaliações",
        coluna_aprovadas: "Unidades aprovadas",
        coluna_sem_avaliacoes: "Unidades sem avaliações",
    }

    if quantidade_zero > 0:
        contexto_zero = (
            bruto.loc[mascara_zero, list(colunas_contexto)]
            .rename(columns=colunas_contexto)
            .describe()
            .T
            .rename(columns={
                "count": "Quantidade válida",
                "mean": "Média",
                "std": "Desvio padrão",
                "min": "Mínimo",
                "25%": "1º quartil",
                "50%": "Mediana",
                "75%": "3º quartil",
                "max": "Máximo",
            })
        )

        contexto_zero.index.name = "Indicador"
        contexto_zero = contexto_zero.reset_index()
        contexto_zero.insert(0, "Semestre", descricao_semestre)

        tabelas_contexto_academico.append(contexto_zero)

        print("\nContexto acadêmico dos estudantes com nota zero:")
        print(contexto_zero.round(2).to_string(index=False))
    else:
        print("\nNenhum registro com nota zero neste semestre.")

# Consolida os resultados dos dois semestres.
tabela_resumo_zero = pd.DataFrame(resumo_notas_zero)

tabela_zero_classes = pd.concat(
    tabelas_zero_por_classe,
    ignore_index=True,
)

print("\n" + "=" * 70)
print("RESUMO DOS CRUZAMENTOS")
print("=" * 70)
print(tabela_resumo_zero.round(2).to_string(index=False))

# Verifica quantos estudantes têm nota zero nos dois semestres.
zero_primeiro = bruto[
    "Curricular units 1st sem (grade)"
].eq(0)

zero_segundo = bruto[
    "Curricular units 2nd sem (grade)"
].eq(0)

print(
    "\nEstudantes com nota zero nos dois semestres:",
    int((zero_primeiro & zero_segundo).sum()),
)
print(
    "Estudantes com nota zero em pelo menos um semestre:",
    int((zero_primeiro | zero_segundo).sum()),
)

# Salva as tabelas para consulta e uso no relatório.
arquivo_resumo_zero = (
    PASTA_RESULTADOS / "resumo_investigacao_notas_zero.csv"
)

arquivo_zero_classes = (
    PASTA_RESULTADOS / "notas_zero_por_classe.csv"
)

tabela_resumo_zero.to_csv(
    arquivo_resumo_zero,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    decimal=",",
    float_format="%.4f",
)

tabela_zero_classes.to_csv(
    arquivo_zero_classes,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    decimal=",",
    float_format="%.4f",
)

print("\nResumo salvo em:", arquivo_resumo_zero)
print("Tabela por classe salva em:", arquivo_zero_classes)

if tabelas_contexto_academico:
    tabela_contexto_zero = pd.concat(
        tabelas_contexto_academico,
        ignore_index=True,
    )

    arquivo_contexto_zero = (
        PASTA_RESULTADOS / "contexto_academico_notas_zero.csv"
    )

    tabela_contexto_zero.to_csv(
        arquivo_contexto_zero,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        float_format="%.4f",
    )

    print("Contexto acadêmico salvo em:", arquivo_contexto_zero)

print(
    "\nInterpretação: os cruzamentos mostram associações, "
    "mas não comprovam a causa das notas zero."
)
print(
    "O percentual por classe representa a proporção de "
    "estudantes daquela classe que possuem nota zero."
)
print(
    "Nenhum zero foi convertido em ausente e "
    "nenhum registro foi removido."
)

print("\nBloco 7 concluído.")

# %% ==========================================================
# BLOCO 8 — Correlações entre variáveis quantitativas
# =============================================================

import numpy as np
import seaborn as sns

print("\n" + "=" * 70)
print("BLOCO 8 — CORRELAÇÕES ENTRE VARIÁVEIS QUANTITATIVAS")
print("=" * 70)

# Mesmas variáveis quantitativas selecionadas no Bloco 5.
dados_correlacao = bruto[VARIAVEIS_QUANTITATIVAS]

# Spearman mede associação monotônica usando os postos dos valores.
matriz_correlacao = dados_correlacao.corr(method="spearman")

# Nomes curtos para facilitar a leitura do gráfico.
ROTULOS_CORRELACAO = {
    "Age at enrollment": "Idade na matrícula",
    "Curricular units 1st sem (credited)": "1º sem. — creditadas",
    "Curricular units 1st sem (enrolled)": "1º sem. — inscritas",
    "Curricular units 1st sem (evaluations)": "1º sem. — avaliações",
    "Curricular units 1st sem (approved)": "1º sem. — aprovadas",
    "Curricular units 1st sem (grade)": "1º sem. — nota média",
    "Curricular units 1st sem (without evaluations)": (
        "1º sem. — sem avaliações"
    ),
    "Curricular units 2nd sem (credited)": "2º sem. — creditadas",
    "Curricular units 2nd sem (enrolled)": "2º sem. — inscritas",
    "Curricular units 2nd sem (evaluations)": "2º sem. — avaliações",
    "Curricular units 2nd sem (approved)": "2º sem. — aprovadas",
    "Curricular units 2nd sem (grade)": "2º sem. — nota média",
    "Curricular units 2nd sem (without evaluations)": (
        "2º sem. — sem avaliações"
    ),
    "Unemployment rate": "Taxa de desemprego",
    "Inflation rate": "Taxa de inflação",
    "GDP": "GDP",
}

print("\nMétodo utilizado: Spearman")
print("Quantidade de variáveis:", len(VARIAVEIS_QUANTITATIVAS))

# Detecta variáveis para as quais a correlação não é definida.
variaveis_constantes = dados_correlacao.columns[
    dados_correlacao.nunique(dropna=True) <= 1
].tolist()

if variaveis_constantes:
    print(
        "\nAtenção: variáveis constantes ou sem dados suficientes:",
        ", ".join(variaveis_constantes),
    )
    print(
        "As correlações dessas variáveis podem aparecer "
        "como células vazias."
    )

# Lista cada par uma única vez, sem a diagonal da matriz.
pares_correlacao = []

for i, variavel_a in enumerate(VARIAVEIS_QUANTITATIVAS):
    for j in range(i + 1, len(VARIAVEIS_QUANTITATIVAS)):
        variavel_b = VARIAVEIS_QUANTITATIVAS[j]
        coeficiente = matriz_correlacao.loc[
            variavel_a, variavel_b
        ]

        if pd.notna(coeficiente):
            pares_correlacao.append({
                "Variável A": variavel_a,
                "Variável B": variavel_b,
                "Correlação de Spearman": coeficiente,
                "Correlação absoluta": abs(coeficiente),
            })

tabela_pares_correlacao = pd.DataFrame(
    pares_correlacao,
    columns=[
        "Variável A",
        "Variável B",
        "Correlação de Spearman",
        "Correlação absoluta",
    ],
)

tabela_pares_correlacao = (
    tabela_pares_correlacao
    .sort_values(
        by="Correlação absoluta",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\nDez pares com maior correlação em valor absoluto:")

if tabela_pares_correlacao.empty:
    print("Não há pares com correlação definida.")
else:
    print(
        tabela_pares_correlacao.head(10).round(3).to_string(
            index=False
        )
    )

# Mantém os nomes originais no CSV e usa traduções no gráfico.
matriz_visual = matriz_correlacao.rename(
    index=ROTULOS_CORRELACAO,
    columns=ROTULOS_CORRELACAO,
)

# Oculta a parte superior porque ela repete a parte inferior.
mascara_superior = np.triu(
    np.ones(matriz_visual.shape, dtype=bool),
    k=1,
)

# Rodapé em uma área própria para evitar sobreposição.
fig_correlacao = plt.figure(
    figsize=(13, 10),
    layout="constrained",
)

fig_correlacao.canvas.manager.set_window_title(
    "Figura 3 — Correlações entre variáveis quantitativas"
)

grade_correlacao = fig_correlacao.add_gridspec(
    nrows=2,
    ncols=1,
    height_ratios=[1, 0.09],
)

ax_correlacao = fig_correlacao.add_subplot(
    grade_correlacao[0, 0]
)

sns.heatmap(
    matriz_visual,
    mask=mascara_superior,
    ax=ax_correlacao,
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    center=0,
    annot=True,
    fmt=".2f",
    annot_kws={"fontsize": 7},
    linewidths=0.4,
    cbar_kws={
        "label": "Correlação de Spearman",
        "shrink": 0.8,
    },
)

ax_correlacao.set_title(
    "Correlações entre variáveis quantitativas — Spearman",
    fontsize=14,
    pad=14,
)

ax_correlacao.set_xlabel("")
ax_correlacao.set_ylabel("")

plt.setp(
    ax_correlacao.get_xticklabels(),
    rotation=45,
    ha="right",
    fontsize=8,
)

plt.setp(
    ax_correlacao.get_yticklabels(),
    rotation=0,
    fontsize=8,
)

ax_fonte_correlacao = fig_correlacao.add_subplot(
    grade_correlacao[1, 0]
)
ax_fonte_correlacao.set_axis_off()

ax_fonte_correlacao.text(
    0.5,
    0.5,
    "Fonte: dataset.csv — elaboração do grupo.\n"
    "Análise exploratória. Correlação não comprova causalidade.\n"
    "Target e códigos categóricos não foram incluídos.",
    transform=ax_fonte_correlacao.transAxes,
    ha="center",
    va="center",
    fontsize=9,
)

arquivo_matriz_correlacao = (
    PASTA_RESULTADOS / "matriz_correlacao_spearman.csv"
)

arquivo_pares_correlacao = (
    PASTA_RESULTADOS / "pares_correlacao_spearman.csv"
)

arquivo_figura_correlacao = (
    PASTA_RESULTADOS / "figura_03_correlacoes_spearman.png"
)

matriz_correlacao.to_csv(
    arquivo_matriz_correlacao,
    sep=";",
    encoding="utf-8-sig",
    decimal=",",
    float_format="%.4f",
)

tabela_pares_correlacao.to_csv(
    arquivo_pares_correlacao,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    decimal=",",
    float_format="%.4f",
)

fig_correlacao.savefig(
    arquivo_figura_correlacao,
    dpi=300,
    bbox_inches="tight",
)

print("\nMatriz salva em:", arquivo_matriz_correlacao)
print("Tabela de pares salva em:", arquivo_pares_correlacao)
print("Figura salva em:", arquivo_figura_correlacao)

print(
    "\nCorrelações altas podem indicar informações semelhantes, "
    "mas não justificam excluir variáveis automaticamente."
)
print(
    "Esta análise é exploratória. Decisões de seleção de "
    "variáveis deverão ser feitas somente com os dados de treino."
)
print("\nNenhum dado foi alterado.")
print("\nFeche a Figura 3 para concluir o Bloco 8.")

plt.show()
plt.close(fig_correlacao)

print("\nBloco 8 concluído.")

# %% ==========================================================
# BLOCO 9 — Divisão aleatória estratificada de treino e teste
# =============================================================

from sklearn.model_selection import train_test_split

print("\n" + "=" * 70)
print("BLOCO 9 — DIVISÃO DE TREINO E TESTE")
print("=" * 70)

SEMENTE_ALEATORIA = 42
PROPORCAO_TESTE = 0.30

# X contém as variáveis de entrada.
# y contém a situação acadêmica que desejamos prever.
# Cópias preservam o DataFrame original.
X = bruto.drop(columns=[ALVO]).copy()
y = bruto[ALVO].copy()

# A divisão é aleatória, reproduzível e estratificada por classe.
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=PROPORCAO_TESTE,
    random_state=SEMENTE_ALEATORIA,
    shuffle=True,
    stratify=y,
)

# Verificações de integridade da divisão.
if not X.index.is_unique:
    raise ValueError(
        "O índice original precisa ser único para conferir a divisão."
    )

if not X_treino.index.equals(y_treino.index):
    raise ValueError(
        "Os índices das entradas e do alvo de treino não coincidem."
    )

if not X_teste.index.equals(y_teste.index):
    raise ValueError(
        "Os índices das entradas e do alvo de teste não coincidem."
    )

indices_compartilhados = X_treino.index.intersection(
    X_teste.index
)

if len(indices_compartilhados) > 0:
    raise ValueError(
        "Foram encontrados registros presentes no treino e no teste."
    )

indices_reunidos = X_treino.index.union(X_teste.index)

if set(indices_reunidos) != set(X.index):
    raise ValueError(
        "A divisão não preservou todos os registros originais."
    )

if ALVO in X_treino.columns or ALVO in X_teste.columns:
    raise ValueError(
        "A variável alvo não pode estar nas variáveis de entrada."
    )

print("\nSemente aleatória:", SEMENTE_ALEATORIA)
print("Estratificação: situação acadêmica (Target)")
print("Total de registros:", len(X))
print("Quantidade de variáveis de entrada:", X.shape[1])

print(
    "\nTreino:",
    len(X_treino),
    f"registros ({len(X_treino) / len(X) * 100:.2f}%)",
)

print(
    "Teste:",
    len(X_teste),
    f"registros ({len(X_teste) / len(X) * 100:.2f}%)",
)

# Compara a distribuição das classes nos três conjuntos.
linhas_divisao = []

for nome_conjunto, alvo_conjunto in [
    ("Original", y),
    ("Treino", y_treino),
    ("Teste", y_teste),
]:
    contagens_conjunto = (
        alvo_conjunto
        .value_counts()
        .reindex(ordem_classes, fill_value=0)
    )

    for classe in ordem_classes:
        quantidade = int(contagens_conjunto.loc[classe])

        linhas_divisao.append({
            "Conjunto": nome_conjunto,
            "Classe": classe,
            "Quantidade": quantidade,
            "Percentual no conjunto (%)": (
                quantidade / len(alvo_conjunto) * 100
            ),
        })

tabela_divisao = pd.DataFrame(linhas_divisao)

print("\nDistribuição das classes:")
print(tabela_divisao.round(2).to_string(index=False))

# Registra a qual conjunto pertence cada linha do CSV.
# O número da linha é uma referência ao arquivo atual,
# não um identificador permanente do estudante.
particao_registros = pd.Series(
    index=bruto.index,
    dtype="object",
)

particao_registros.loc[X_treino.index] = "Treino"
particao_registros.loc[X_teste.index] = "Teste"

tabela_particao = pd.DataFrame({
    "Indice original": bruto.index,
    "Classe": y.to_numpy(),
    "Conjunto": particao_registros.to_numpy(),
})

arquivo_distribuicao_divisao = (
    PASTA_RESULTADOS / "distribuicao_classes_treino_teste.csv"
)

arquivo_particao = (
    PASTA_RESULTADOS / "particao_registros_treino_teste.csv"
)

tabela_divisao.to_csv(
    arquivo_distribuicao_divisao,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    decimal=",",
    float_format="%.4f",
)

tabela_particao.to_csv(
    arquivo_particao,
    index=False,
    sep=";",
    encoding="utf-8-sig",
)

print("\nDistribuição salva em:", arquivo_distribuicao_divisao)
print("Identificação da partição salva em:", arquivo_particao)

print("\nVerificações concluídas:")
print("- Entradas e alvo estão alinhados.")
print("- Nenhum registro pertence simultaneamente a treino e teste.")
print("- Todos os registros originais foram preservados.")
print("- Target foi separado das variáveis de entrada.")

print(
    "\nPré-processamento e ajuste de hiperparâmetros "
    "deverão usar somente o treino."
)
print(
    "O teste ficará reservado para a avaliação final "
    "dos modelos definidos com os dados de treino."
)
print(
    "Esta divisão pressupõe registros independentes. "
    "Ela não representa uma separação temporal."
)
print(
    "Como usamos informações dos dois semestres, "
    "o cenário de previsão é posterior à disponibilidade "
    "desses dados, e não o instante da matrícula."
)

print("\nNenhum modelo foi treinado neste bloco.")
print("\nBloco 9 concluído.")

# %% ==========================================================
# BLOCO 9A — Desempenho acadêmico por classe no treino
# =============================================================

import matplotlib.pyplot as plt

print("\n" + "=" * 70)
print("BLOCO 9A — DESEMPENHO ACADÊMICO POR CLASSE NO TREINO")
print("=" * 70)

variaveis_9a = [
    ("Curricular units 1st sem (grade)", "Nota média — 1º semestre", "Nota"),
    ("Curricular units 2nd sem (grade)", "Nota média — 2º semestre", "Nota"),
    ("Curricular units 1st sem (approved)", "Aprovações — 1º semestre", "Disciplinas aprovadas"),
    ("Curricular units 2nd sem (approved)", "Aprovações — 2º semestre", "Disciplinas aprovadas"),
]

classes_9a = ["Dropout", "Enrolled", "Graduate"]
rotulos_9a = ["Evasão", "Matriculado", "Formado"]
cores_9a = ["#C96C56", "#D7AB4C", "#4F8C79"]

colunas_9a = [coluna for coluna, _, _ in variaveis_9a]
dados_9a = X_treino[colunas_9a].assign(**{ALVO: y_treino})
grupos_9a = dados_9a.groupby(ALVO, sort=False)

contagens_9a = grupos_9a.size().reindex(classes_9a)
rotulos_com_contagem_9a = [
    f"{rotulo}\n(n = {contagens_9a.loc[classe]})"
    for classe, rotulo in zip(classes_9a, rotulos_9a)
]

resumo_9a = grupos_9a[colunas_9a].median().reindex(classes_9a)
resumo_9a = resumo_9a.rename(
    columns={
        coluna: f"Mediana — {titulo}"
        for coluna, titulo, _ in variaveis_9a
    }
)
resumo_9a.insert(0, "Estudantes", contagens_9a)
resumo_9a.insert(0, "Situação acadêmica", rotulos_9a)
resumo_9a = resumo_9a.reset_index()

figura_9a, eixos_9a = plt.subplots(
    2, 2, figsize=(12, 8), constrained_layout=True
)
figura_9a.suptitle(
    f"Desempenho acadêmico por classe no treino (n = {len(dados_9a)})",
    fontsize=15,
)

for eixo, (coluna, titulo, unidade) in zip(eixos_9a.flat, variaveis_9a):
    valores_por_classe = [
        grupos_9a.get_group(classe)[coluna].to_numpy()
        for classe in classes_9a
    ]

    caixas = eixo.boxplot(
        valores_por_classe,
        patch_artist=True,
        medianprops={"color": "#1F2937", "linewidth": 2},
        flierprops={"markersize": 2, "alpha": 0.35},
    )

    for caixa, cor in zip(caixas["boxes"], cores_9a):
        caixa.set_facecolor(cor)
        caixa.set_alpha(0.75)

    eixo.set_xticks([1, 2, 3], rotulos_com_contagem_9a)
    eixo.set_title(titulo)
    eixo.set_ylabel(unidade)
    eixo.grid(axis="y", alpha=0.25)
    eixo.set_axisbelow(True)

pasta_saidas_9a = PASTA_PROJETO / "resultados"
pasta_saidas_9a.mkdir(parents=True, exist_ok=True)

arquivo_figura_9a = pasta_saidas_9a / "figura_08_desempenho_por_classe_treino.png"
arquivo_resumo_9a = pasta_saidas_9a / "resumo_desempenho_por_classe_treino.csv"

figura_9a.savefig(arquivo_figura_9a, dpi=180, bbox_inches="tight")
plt.close(figura_9a)

resumo_9a.to_csv(
    arquivo_resumo_9a,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    decimal=",",
    float_format="%.2f",
)

print(f"\nRegistros de treino analisados: {len(dados_9a)}")
print(resumo_9a.round(2).to_string(index=False))
print(f"\nFigura salva em: {arquivo_figura_9a}")
print(f"Tabela salva em: {arquivo_resumo_9a}")
print("\nBloco 9A concluído.")


# %% ==========================================================
# BLOCO 10 — Preparação do pré-processamento
# =============================================================

import sklearn

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("\n" + "=" * 70)
print("BLOCO 10 — PREPARAÇÃO DO PRÉ-PROCESSAMENTO")
print("=" * 70)

print("\nVersão do scikit-learn:", sklearn.__version__)

# As 16 medidas do Bloco 5 mais a ordem da candidatura.
# Nesta abordagem, a ordem é tratada como ordinal numérica.
COLUNAS_NUMERICAS = [
    *VARIAVEIS_QUANTITATIVAS,
    "Application order",
]

# Códigos representam categorias, não quantidades.
# Mantém-se "Nacionality", conforme o nome existente no CSV.
COLUNAS_CATEGORICAS = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Gender",
]

# Indicadores que devem estar codificados como 0 e 1.
COLUNAS_BINARIAS = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
    "International",
]

# Confere se cada entrada pertence a exatamente um grupo.
colunas_classificadas = (
    COLUNAS_NUMERICAS
    + COLUNAS_CATEGORICAS
    + COLUNAS_BINARIAS
)

if len(colunas_classificadas) != len(set(colunas_classificadas)):
    raise ValueError(
        "Uma coluna foi incluída em mais de um grupo."
    )

colunas_faltantes = (
    set(X_treino.columns) - set(colunas_classificadas)
)

colunas_inexistentes = (
    set(colunas_classificadas) - set(X_treino.columns)
)

if colunas_faltantes or colunas_inexistentes:
    raise ValueError(
        "A classificação das colunas está incompleta ou incorreta.\n"
        f"Sem grupo: {sorted(colunas_faltantes)}\n"
        f"Inexistentes: {sorted(colunas_inexistentes)}"
    )

if ALVO in colunas_classificadas:
    raise ValueError(
        "Target não pode fazer parte do pré-processamento."
    )

# Verifica somente o treino.
if X_treino[colunas_classificadas].isna().any().any():
    raise ValueError(
        "Foram encontrados valores ausentes no treino. "
        "É necessário revisar o tratamento antes de continuar."
    )

for coluna in COLUNAS_NUMERICAS + COLUNAS_BINARIAS:
    if not pd.api.types.is_numeric_dtype(X_treino[coluna]):
        raise TypeError(
            f"A coluna '{coluna}' precisa ser numérica."
        )

    if not np.isfinite(X_treino[coluna].to_numpy()).all():
        raise ValueError(
            f"A coluna '{coluna}' contém valores não finitos."
        )

for coluna in COLUNAS_BINARIAS:
    valores = set(X_treino[coluna].unique())

    if not valores.issubset({0, 1}):
        raise ValueError(
            f"O indicador '{coluna}' deveria conter somente 0 e 1. "
            f"Valores encontrados: {sorted(valores)}"
        )

# Cria o objeto, mas não aprende categorias nem estatísticas.
preprocessador = ColumnTransformer(
    transformers=[
        (
            "numericas",
            StandardScaler(),
            COLUNAS_NUMERICAS,
        ),
        (
            "categoricas",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            COLUNAS_CATEGORICAS,
        ),
        (
            "binarias",
            "passthrough",
            COLUNAS_BINARIAS,
        ),
    ],
    remainder="drop",
)

# Cada modelo receberá uma cópia independente do preprocessador.
# Esta função apenas monta o Pipeline; não executa treinamento.
def criar_pipeline(classificador):
    return Pipeline(
        steps=[
            ("preprocessamento", clone(preprocessador)),
            ("modelo", clone(classificador)),
        ]
    )


# Documenta a decisão de tratamento de cada entrada.
linhas_preprocessamento = []

for coluna in COLUNAS_NUMERICAS:
    linhas_preprocessamento.append({
        "Variável": coluna,
        "Grupo": (
            "Ordinal numérica"
            if coluna == "Application order"
            else "Quantitativa"
        ),
        "Tratamento": "StandardScaler",
    })

for coluna in COLUNAS_CATEGORICAS:
    linhas_preprocessamento.append({
        "Variável": coluna,
        "Grupo": "Categórica",
        "Tratamento": "OneHotEncoder",
    })

for coluna in COLUNAS_BINARIAS:
    linhas_preprocessamento.append({
        "Variável": coluna,
        "Grupo": "Binária",
        "Tratamento": "Preservar 0 e 1",
    })

tabela_preprocessamento = pd.DataFrame(
    linhas_preprocessamento
)

print("\nPlano de pré-processamento:")
print(tabela_preprocessamento.to_string(index=False))

print("\nQuantidade de entradas por grupo:")
print("Numéricas, incluindo ordem da candidatura:", len(COLUNAS_NUMERICAS))
print("Categóricas:", len(COLUNAS_CATEGORICAS))
print("Binárias:", len(COLUNAS_BINARIAS))
print("Total:", len(colunas_classificadas))

arquivo_preprocessamento = (
    PASTA_RESULTADOS / "plano_preprocessamento.csv"
)

tabela_preprocessamento.to_csv(
    arquivo_preprocessamento,
    index=False,
    sep=";",
    encoding="utf-8-sig",
)

print("\nPlano salvo em:", arquivo_preprocessamento)

print("\nVerificações concluídas:")
print("- Todas as entradas pertencem a exatamente um grupo.")
print("- Target ficou fora do pré-processamento.")
print("- Indicadores binários do treino contêm somente 0 e 1.")
print("- Nenhum valor ausente foi encontrado no treino.")

print(
    "\nO preprocessador foi criado, mas ainda não foi ajustado."
)
print(
    "Na validação cruzada, o Pipeline aprenderá a padronização "
    "e as categorias somente na parte de treino de cada divisão."
)
print(
    "Categorias desconhecidas serão representadas por zeros "
    "nas colunas correspondentes àquela variável."
)
print(
    "Os zeros originais das notas foram preservados. "
    "Nenhuma imputação ou remoção de registros foi aplicada."
)
print(
    "A quantidade de colunas após a codificação será conhecida "
    "quando o preprocessador for ajustado."
)
print(
    "Este é o plano inicial. Eventuais alternativas serão "
    "comparadas usando somente o treino."
)

print("\nNenhum modelo foi treinado e o teste não foi transformado.")
print("\nBloco 10 concluído.")

# %% ==========================================================
# BLOCO 11 — Referência inicial e validação cruzada
# =============================================================

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)

print("\n" + "=" * 70)
print("BLOCO 11 — REFERÊNCIA INICIAL E VALIDAÇÃO CRUZADA")
print("=" * 70)

NUMERO_DIVISOES_CV = 5

# Verifica se cada classe possui registros suficientes.
contagens_treino_cv = (
    y_treino.value_counts()
    .reindex(ordem_classes, fill_value=0)
)

if contagens_treino_cv.min() < NUMERO_DIVISOES_CV:
    raise ValueError(
        "Cada classe precisa ter pelo menos cinco registros "
        "para esta validação cruzada."
    )

# A mesma configuração será usada nos classificadores seguintes.
validacao_cruzada = StratifiedKFold(
    n_splits=NUMERO_DIVISOES_CV,
    shuffle=True,
    random_state=SEMENTE_ALEATORIA,
)

# Materializa as divisões para reutilizá-las em todos os modelos.
# Os índices são posições dentro do conjunto de treino.
divisoes_cv = list(
    validacao_cruzada.split(X_treino, y_treino)
)

metricas_cv = {
    "acuracia": "accuracy",
    "precisao_macro": make_scorer(
        precision_score,
        average="macro",
        labels=ordem_classes,
        zero_division=0,
    ),
    "recall_macro": make_scorer(
        recall_score,
        average="macro",
        labels=ordem_classes,
        zero_division=0,
    ),
    "f1_macro": make_scorer(
        f1_score,
        average="macro",
        labels=ordem_classes,
        zero_division=0,
    ),
}

# Referência simples: sempre prevê a classe mais frequente.
# O Dummy não precisa de pré-processamento porque ignora
# os valores das entradas para definir suas previsões.
modelo_referencia = DummyClassifier(
    strategy="most_frequent",
    random_state=SEMENTE_ALEATORIA,
)

print("\nModelo: DummyClassifier")
print("Estratégia: prever sempre a classe mais frequente")
print("Quantidade de divisões:", NUMERO_DIVISOES_CV)
print("Registros disponíveis para validação cruzada:", len(X_treino))
print("Classe mais frequente no treino:", y_treino.value_counts().idxmax())
print("\nExecutando validação cruzada...")

resultados_dummy_cv = cross_validate(
    estimator=modelo_referencia,
    X=X_treino,
    y=y_treino,
    cv=divisoes_cv,
    scoring=metricas_cv,
    return_train_score=True,
    n_jobs=1,
    error_score="raise",
)

# Uma linha para cada divisão da validação cruzada.
linhas_dummy_cv = []

for numero, (posicoes_treino, posicoes_validacao) in enumerate(
    divisoes_cv,
    start=1,
):
    indice_resultado = numero - 1

    linha = {
        "Divisão": numero,
        "Registros de treino": len(posicoes_treino),
        "Registros de validação": len(posicoes_validacao),
        "Tempo de ajuste (s)": resultados_dummy_cv[
            "fit_time"
        ][indice_resultado],
        "Tempo de avaliação (s)": resultados_dummy_cv[
            "score_time"
        ][indice_resultado],
    }

    for nome_metrica in metricas_cv:
        linha[f"Treino - {nome_metrica}"] = (
            resultados_dummy_cv[f"train_{nome_metrica}"][
                indice_resultado
            ]
        )
        linha[f"Validação - {nome_metrica}"] = (
            resultados_dummy_cv[f"test_{nome_metrica}"][
                indice_resultado
            ]
        )

    linhas_dummy_cv.append(linha)

tabela_dummy_divisoes = pd.DataFrame(linhas_dummy_cv)

# Resume as métricas entre as cinco divisões.
linhas_resumo_dummy = []

for nome_metrica in metricas_cv:
    valores_treino = resultados_dummy_cv[
        f"train_{nome_metrica}"
    ]
    valores_validacao = resultados_dummy_cv[
        f"test_{nome_metrica}"
    ]

    linhas_resumo_dummy.append({
        "Modelo": "Dummy — classe mais frequente",
        "Métrica": nome_metrica,
        "Média no treino": np.mean(valores_treino),
        "Desvio padrão no treino": np.std(
            valores_treino,
            ddof=1,
        ),
        "Média na validação": np.mean(valores_validacao),
        "Desvio padrão na validação": np.std(
            valores_validacao,
            ddof=1,
        ),
        "Diferença treino - validação": (
            np.mean(valores_treino)
            - np.mean(valores_validacao)
        ),
    })

tabela_dummy_resumo = pd.DataFrame(linhas_resumo_dummy)

tabela_dummy_tempos = pd.DataFrame([{
    "Modelo": "Dummy — classe mais frequente",
    "Divisões": NUMERO_DIVISOES_CV,
    "Tempo médio de ajuste (s)": np.mean(
        resultados_dummy_cv["fit_time"]
    ),
    "Tempo médio de avaliação (s)": np.mean(
        resultados_dummy_cv["score_time"]
    ),
}])

print("\nResultados por divisão:")
print(tabela_dummy_divisoes.round(4).to_string(index=False))

print("\nResumo das métricas:")
print(tabela_dummy_resumo.round(4).to_string(index=False))

print("\nTempos médios:")
print(tabela_dummy_tempos.round(6).to_string(index=False))

arquivo_dummy_divisoes = (
    PASTA_RESULTADOS / "dummy_validacao_cruzada_divisoes.csv"
)

arquivo_dummy_resumo = (
    PASTA_RESULTADOS / "dummy_validacao_cruzada_resumo.csv"
)

arquivo_dummy_tempos = (
    PASTA_RESULTADOS / "dummy_validacao_cruzada_tempos.csv"
)

tabela_dummy_divisoes.to_csv(
    arquivo_dummy_divisoes,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    decimal=",",
    float_format="%.6f",
)

tabela_dummy_resumo.to_csv(
    arquivo_dummy_resumo,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    decimal=",",
    float_format="%.6f",
)

tabela_dummy_tempos.to_csv(
    arquivo_dummy_tempos,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    decimal=",",
    float_format="%.6f",
)

print("\nResultados por divisão salvos em:", arquivo_dummy_divisoes)
print("Resumo salvo em:", arquivo_dummy_resumo)
print("Tempos salvos em:", arquivo_dummy_tempos)

print(
    "\nAs métricas estão na escala de 0 a 1. "
    "Multiplique por 100 para expressá-las em percentual."
)
print(
    "A média macro atribui o mesmo peso às três classes."
)
print(
    "Quando uma classe não recebe previsões, "
    "zero_division=0 atribui zero à precisão indefinida."
)
print(
    "As chaves 'test_' retornadas pelo cross_validate "
    "representam a validação de cada divisão do treino. "
    "Não representam o conjunto de teste reservado."
)
print(
    "O tempo de avaliação inclui as previsões e o cálculo "
    "das métricas; não mede apenas a previsão."
)
print(
    "Os desvios padrão descrevem a variação entre as divisões; "
    "não são intervalos de confiança."
)
print(
    "Esta referência não conta como um dos seis "
    "algoritmos exigidos no projeto."
)

print("\nO conjunto de teste reservado não foi utilizado.")
print("\nBloco 11 concluído.")

# %% ==========================================================
# BLOCO 12 — Regressão Logística inicial
# =============================================================

import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

print("\n" + "=" * 70)
print("BLOCO 12 — REGRESSÃO LOGÍSTICA INICIAL")
print("=" * 70)

# Configuração inicial, ainda sem busca de hiperparâmetros.
# C controla a regularização: valores menores aumentam sua força.
# max_iter define o limite de iterações do otimizador.
classificador_logistica = LogisticRegression(
    solver="lbfgs",
    C=1.0,
    max_iter=3000,
    class_weight=None,
    random_state=SEMENTE_ALEATORIA,
)

pipeline_logistica = criar_pipeline(
    classificador_logistica
)

print("\nModelo: Regressão Logística")
print("Solver: lbfgs")
print("C: 1.0")
print("Limite de iterações: 3000")
print("Pesos das classes: sem ponderação adicional")
print("Divisões da validação cruzada:", len(divisoes_cv))
print("Registros disponíveis:", len(X_treino))

print(
    "\nO pré-processamento será ajustado separadamente "
    "dentro de cada divisão."
)
print(
    "Executando validação cruzada. Aguarde a conclusão..."
)

# Transforma falha de convergência em erro visível.
# Isso evita aceitar silenciosamente um ajuste incompleto.
with warnings.catch_warnings():
    warnings.simplefilter("error", ConvergenceWarning)

    resultados_logistica_cv = cross_validate(
        estimator=pipeline_logistica,
        X=X_treino,
        y=y_treino,
        cv=divisoes_cv,
        scoring=metricas_cv,
        return_train_score=True,
        n_jobs=1,
        error_score="raise",
    )

# Resultados individuais das cinco divisões.
linhas_logistica_cv = []

for numero, (posicoes_treino, posicoes_validacao) in enumerate(
    divisoes_cv,
    start=1,
):
    indice_resultado = numero - 1

    linha = {
        "Divisão": numero,
        "Registros de treino": len(posicoes_treino),
        "Registros de validação": len(posicoes_validacao),
        "Tempo de ajuste (s)": resultados_logistica_cv[
            "fit_time"
        ][indice_resultado],
        "Tempo de avaliação (s)": resultados_logistica_cv[
            "score_time"
        ][indice_resultado],
    }

    for nome_metrica in metricas_cv:
        linha[f"Treino - {nome_metrica}"] = (
            resultados_logistica_cv[f"train_{nome_metrica}"][
                indice_resultado
            ]
        )

        linha[f"Validação - {nome_metrica}"] = (
            resultados_logistica_cv[f"test_{nome_metrica}"][
                indice_resultado
            ]
        )

    linhas_logistica_cv.append(linha)

tabela_logistica_divisoes = pd.DataFrame(
    linhas_logistica_cv
)

# Resumo das métricas entre as divisões.
linhas_resumo_logistica = []

for nome_metrica in metricas_cv:
    valores_treino = resultados_logistica_cv[
        f"train_{nome_metrica}"
    ]

    valores_validacao = resultados_logistica_cv[
        f"test_{nome_metrica}"
    ]

    linhas_resumo_logistica.append({
        "Modelo": "Regressão Logística — inicial",
        "Métrica": nome_metrica,
        "Média no treino": np.mean(valores_treino),
        "Desvio padrão no treino": np.std(
            valores_treino,
            ddof=1,
        ),
        "Média na validação": np.mean(valores_validacao),
        "Desvio padrão na validação": np.std(
            valores_validacao,
            ddof=1,
        ),
        "Diferença treino - validação": (
            np.mean(valores_treino)
            - np.mean(valores_validacao)
        ),
    })

tabela_logistica_resumo = pd.DataFrame(
    linhas_resumo_logistica
)

# O tempo de ajuste inclui pré-processamento e classificador.
tabela_logistica_tempos = pd.DataFrame([{
    "Modelo": "Regressão Logística — inicial",
    "Divisões": len(divisoes_cv),
    "Tempo médio de ajuste (s)": np.mean(
        resultados_logistica_cv["fit_time"]
    ),
    "Tempo médio de avaliação (s)": np.mean(
        resultados_logistica_cv["score_time"]
    ),
}])

# Compara as médias de validação com a referência do Bloco 11.
comparacao_dummy_logistica = (
    tabela_dummy_resumo[
        ["Métrica", "Média na validação"]
    ]
    .rename(columns={
        "Média na validação": "Dummy - média na validação",
    })
    .merge(
        tabela_logistica_resumo[
            ["Métrica", "Média na validação"]
        ].rename(columns={
            "Média na validação": (
                "Logística - média na validação"
            ),
        }),
        on="Métrica",
        validate="one_to_one",
    )
)

comparacao_dummy_logistica[
    "Ganho sobre Dummy (pontos percentuais)"
] = (
    comparacao_dummy_logistica[
        "Logística - média na validação"
    ]
    - comparacao_dummy_logistica[
        "Dummy - média na validação"
    ]
) * 100

print("\nResultados por divisão:")
print(
    tabela_logistica_divisoes.round(4).to_string(
        index=False
    )
)

print("\nResumo das métricas:")
print(
    tabela_logistica_resumo.round(4).to_string(
        index=False
    )
)

print("\nComparação com o Dummy:")
print(
    comparacao_dummy_logistica.round(4).to_string(
        index=False
    )
)

print("\nTempos médios:")
print(
    tabela_logistica_tempos.round(6).to_string(
        index=False
    )
)

arquivo_logistica_divisoes = (
    PASTA_RESULTADOS
    / "logistica_inicial_validacao_cruzada_divisoes.csv"
)

arquivo_logistica_resumo = (
    PASTA_RESULTADOS
    / "logistica_inicial_validacao_cruzada_resumo.csv"
)

arquivo_logistica_tempos = (
    PASTA_RESULTADOS
    / "logistica_inicial_validacao_cruzada_tempos.csv"
)

arquivo_comparacao_dummy_logistica = (
    PASTA_RESULTADOS
    / "comparacao_dummy_logistica_inicial.csv"
)

for tabela, caminho in [
    (tabela_logistica_divisoes, arquivo_logistica_divisoes),
    (tabela_logistica_resumo, arquivo_logistica_resumo),
    (tabela_logistica_tempos, arquivo_logistica_tempos),
    (
        comparacao_dummy_logistica,
        arquivo_comparacao_dummy_logistica,
    ),
]:
    tabela.to_csv(
        caminho,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        float_format="%.6f",
    )

    print("\nTabela salva em:", caminho)

print(
    "\nAs médias das métricas estão na escala de 0 a 1."
)
print(
    "O ganho sobre o Dummy está em pontos percentuais, "
    "não em percentual relativo."
)
print(
    "Diferenças entre treino e validação ajudam a investigar "
    "overfitting, mas não devem ser interpretadas isoladamente."
)
print(
    "O tempo de ajuste inclui o pré-processamento. "
    "O Dummy não precisa dessa etapa."
)
print(
    "O tempo de avaliação inclui transformação, previsões "
    "e cálculo das métricas."
)
print(
    "Esta é uma avaliação inicial, sem busca de hiperparâmetros."
)
print(
    "O cross_validate ajustou cópias do Pipeline. "
    "pipeline_logistica ainda não é um modelo final treinado."
)

print("\nO conjunto de teste reservado não foi utilizado.")
print("\nBloco 12 concluído.")

# %% ==========================================================
# BLOCO 13 — KNN inicial
# =============================================================

from sklearn.neighbors import KNeighborsClassifier

print("\n" + "=" * 70)
print("BLOCO 13 — KNN INICIAL")
print("=" * 70)

# Configuração inicial, sem busca de hiperparâmetros.
classificador_knn = KNeighborsClassifier(
    n_neighbors=5,
    weights="uniform",
    metric="minkowski",
    p=2,
    algorithm="brute",
    n_jobs=1,
)

pipeline_knn = criar_pipeline(classificador_knn)

print("\nModelo: KNN")
print("Quantidade de vizinhos: 5")
print("Votação: pesos iguais")
print("Distância: euclidiana")
print("Busca dos vizinhos: força bruta")
print("Divisões da validação cruzada:", len(divisoes_cv))
print("Registros disponíveis:", len(X_treino))

print(
    "\nAs medidas numéricas serão padronizadas dentro "
    "de cada divisão da validação cruzada."
)
print(
    "Executando validação cruzada. Aguarde a conclusão..."
)

resultados_knn_cv = cross_validate(
    estimator=pipeline_knn,
    X=X_treino,
    y=y_treino,
    cv=divisoes_cv,
    scoring=metricas_cv,
    return_train_score=True,
    n_jobs=1,
    error_score="raise",
)

# Resultados individuais das cinco divisões.
linhas_knn_cv = []

for numero, (posicoes_treino, posicoes_validacao) in enumerate(
    divisoes_cv,
    start=1,
):
    indice_resultado = numero - 1

    linha = {
        "Divisão": numero,
        "Registros de treino": len(posicoes_treino),
        "Registros de validação": len(posicoes_validacao),
        "Tempo de ajuste (s)": resultados_knn_cv[
            "fit_time"
        ][indice_resultado],
        "Tempo de avaliação (s)": resultados_knn_cv[
            "score_time"
        ][indice_resultado],
    }

    for nome_metrica in metricas_cv:
        linha[f"Treino - {nome_metrica}"] = (
            resultados_knn_cv[f"train_{nome_metrica}"][
                indice_resultado
            ]
        )

        linha[f"Validação - {nome_metrica}"] = (
            resultados_knn_cv[f"test_{nome_metrica}"][
                indice_resultado
            ]
        )

    linhas_knn_cv.append(linha)

tabela_knn_divisoes = pd.DataFrame(linhas_knn_cv)

# Resumo das métricas entre as divisões.
linhas_resumo_knn = []

for nome_metrica in metricas_cv:
    valores_treino = resultados_knn_cv[
        f"train_{nome_metrica}"
    ]

    valores_validacao = resultados_knn_cv[
        f"test_{nome_metrica}"
    ]

    linhas_resumo_knn.append({
        "Modelo": "KNN — inicial",
        "Métrica": nome_metrica,
        "Média no treino": np.mean(valores_treino),
        "Desvio padrão no treino": np.std(
            valores_treino,
            ddof=1,
        ),
        "Média na validação": np.mean(valores_validacao),
        "Desvio padrão na validação": np.std(
            valores_validacao,
            ddof=1,
        ),
        "Diferença treino - validação": (
            np.mean(valores_treino)
            - np.mean(valores_validacao)
        ),
    })

tabela_knn_resumo = pd.DataFrame(linhas_resumo_knn)

tabela_knn_tempos = pd.DataFrame([{
    "Modelo": "KNN — inicial",
    "Divisões": len(divisoes_cv),
    "Tempo médio de ajuste (s)": np.mean(
        resultados_knn_cv["fit_time"]
    ),
    "Tempo médio de avaliação (s)": np.mean(
        resultados_knn_cv["score_time"]
    ),
}])

# Reúne as médias de validação dos três modelos já avaliados.
comparacao_inicial_knn = (
    tabela_dummy_resumo[
        ["Métrica", "Média na validação"]
    ]
    .rename(columns={
        "Média na validação": "Dummy",
    })
    .merge(
        tabela_logistica_resumo[
            ["Métrica", "Média na validação"]
        ].rename(columns={
            "Média na validação": "Regressão Logística",
        }),
        on="Métrica",
        validate="one_to_one",
    )
    .merge(
        tabela_knn_resumo[
            ["Métrica", "Média na validação"]
        ].rename(columns={
            "Média na validação": "KNN",
        }),
        on="Métrica",
        validate="one_to_one",
    )
)

comparacao_inicial_knn[
    "KNN - Dummy (pontos percentuais)"
] = (
    comparacao_inicial_knn["KNN"]
    - comparacao_inicial_knn["Dummy"]
) * 100

comparacao_inicial_knn[
    "KNN - Logística (pontos percentuais)"
] = (
    comparacao_inicial_knn["KNN"]
    - comparacao_inicial_knn["Regressão Logística"]
) * 100

print("\nResultados por divisão:")
print(
    tabela_knn_divisoes.round(4).to_string(index=False)
)

print("\nResumo das métricas:")
print(
    tabela_knn_resumo.round(4).to_string(index=False)
)

print("\nComparação das médias na validação:")
print(
    comparacao_inicial_knn.round(4).to_string(index=False)
)

print("\nTempos médios:")
print(
    tabela_knn_tempos.round(6).to_string(index=False)
)

arquivo_knn_divisoes = (
    PASTA_RESULTADOS
    / "knn_inicial_validacao_cruzada_divisoes.csv"
)

arquivo_knn_resumo = (
    PASTA_RESULTADOS
    / "knn_inicial_validacao_cruzada_resumo.csv"
)

arquivo_knn_tempos = (
    PASTA_RESULTADOS
    / "knn_inicial_validacao_cruzada_tempos.csv"
)

arquivo_comparacao_knn = (
    PASTA_RESULTADOS
    / "comparacao_dummy_logistica_knn_iniciais.csv"
)

for tabela, caminho in [
    (tabela_knn_divisoes, arquivo_knn_divisoes),
    (tabela_knn_resumo, arquivo_knn_resumo),
    (tabela_knn_tempos, arquivo_knn_tempos),
    (comparacao_inicial_knn, arquivo_comparacao_knn),
]:
    tabela.to_csv(
        caminho,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        float_format="%.6f",
    )

    print("\nTabela salva em:", caminho)

print(
    "\nAs métricas estão na escala de 0 a 1. "
    "As diferenças estão em pontos percentuais."
)
print(
    "Diferença positiva significa que o KNN teve maior "
    "média naquela métrica; negativa significa menor média."
)
print(
    "O KNN concentra parte importante do trabalho na previsão, "
    "quando procura os vizinhos."
)
print(
    "O tempo de avaliação inclui transformação, previsões "
    "e cálculo das métricas."
)
print(
    "Na avaliação do treino, cada registro também pode ser "
    "seu próprio vizinho. Essa pontuação não substitui "
    "a avaliação nas divisões de validação."
)
print(
    "Esta configuração é inicial. A quantidade de vizinhos, "
    "os pesos e a distância poderão ser ajustados "
    "posteriormente usando somente o treino."
)
print(
    "As médias, isoladamente, não demonstram "
    "significância estatística entre os modelos."
)
print(
    "O cross_validate ajustou cópias do Pipeline. "
    "pipeline_knn ainda não é um modelo final treinado."
)

print("\nO conjunto de teste reservado não foi utilizado.")
print("\nBloco 13 concluído.")

# %% ==========================================================
# BLOCO 14 — Árvore de Decisão inicial
# =============================================================

from sklearn.tree import DecisionTreeClassifier

print("\n" + "=" * 70)
print("BLOCO 14 — ÁRVORE DE DECISÃO INICIAL")
print("=" * 70)

# Configuração inicial, sem busca de hiperparâmetros.
classificador_arvore = DecisionTreeClassifier(
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight=None,
    random_state=SEMENTE_ALEATORIA,
)

pipeline_arvore = criar_pipeline(classificador_arvore)

print("\nModelo: Árvore de Decisão")
print("Critério de divisão: Gini")
print("Profundidade máxima: sem limite")
print("Mínimo de registros para dividir um nó: 2")
print("Mínimo de registros por folha: 1")
print("Pesos das classes: sem ponderação adicional")
print("Divisões da validação cruzada:", len(divisoes_cv))
print("Registros disponíveis:", len(X_treino))

print(
    "\nO pré-processamento será ajustado dentro de cada divisão."
)
print(
    "A árvore não exige padronização, mas manteremos "
    "o Pipeline comum nesta comparação inicial."
)
print(
    "\nExecutando validação cruzada. Aguarde a conclusão..."
)

resultados_arvore_cv = cross_validate(
    estimator=pipeline_arvore,
    X=X_treino,
    y=y_treino,
    cv=divisoes_cv,
    scoring=metricas_cv,
    return_train_score=True,
    return_estimator=True,
    n_jobs=1,
    error_score="raise",
)

# Resultados e tamanho da árvore de cada divisão.
linhas_arvore_cv = []

for numero, (posicoes_treino, posicoes_validacao) in enumerate(
    divisoes_cv,
    start=1,
):
    indice_resultado = numero - 1

    arvore_ajustada = resultados_arvore_cv[
        "estimator"
    ][indice_resultado].named_steps["modelo"]

    linha = {
        "Divisão": numero,
        "Registros de treino": len(posicoes_treino),
        "Registros de validação": len(posicoes_validacao),
        "Profundidade da árvore": arvore_ajustada.get_depth(),
        "Quantidade de folhas": arvore_ajustada.get_n_leaves(),
        "Tempo de ajuste (s)": resultados_arvore_cv[
            "fit_time"
        ][indice_resultado],
        "Tempo de avaliação (s)": resultados_arvore_cv[
            "score_time"
        ][indice_resultado],
    }

    for nome_metrica in metricas_cv:
        linha[f"Treino - {nome_metrica}"] = (
            resultados_arvore_cv[f"train_{nome_metrica}"][
                indice_resultado
            ]
        )

        linha[f"Validação - {nome_metrica}"] = (
            resultados_arvore_cv[f"test_{nome_metrica}"][
                indice_resultado
            ]
        )

    linhas_arvore_cv.append(linha)

tabela_arvore_divisoes = pd.DataFrame(linhas_arvore_cv)

# Resumo das métricas entre as divisões.
linhas_resumo_arvore = []

for nome_metrica in metricas_cv:
    valores_treino = resultados_arvore_cv[
        f"train_{nome_metrica}"
    ]

    valores_validacao = resultados_arvore_cv[
        f"test_{nome_metrica}"
    ]

    linhas_resumo_arvore.append({
        "Modelo": "Árvore de Decisão — inicial",
        "Métrica": nome_metrica,
        "Média no treino": np.mean(valores_treino),
        "Desvio padrão no treino": np.std(
            valores_treino,
            ddof=1,
        ),
        "Média na validação": np.mean(valores_validacao),
        "Desvio padrão na validação": np.std(
            valores_validacao,
            ddof=1,
        ),
        "Diferença treino - validação": (
            np.mean(valores_treino)
            - np.mean(valores_validacao)
        ),
    })

tabela_arvore_resumo = pd.DataFrame(linhas_resumo_arvore)

tabela_arvore_tempos = pd.DataFrame([{
    "Modelo": "Árvore de Decisão — inicial",
    "Divisões": len(divisoes_cv),
    "Tempo médio de ajuste (s)": np.mean(
        resultados_arvore_cv["fit_time"]
    ),
    "Tempo médio de avaliação (s)": np.mean(
        resultados_arvore_cv["score_time"]
    ),
}])

# Comparação com todos os modelos iniciais já avaliados.
comparacao_inicial_arvore = (
    tabela_dummy_resumo[
        ["Métrica", "Média na validação"]
    ]
    .rename(columns={
        "Média na validação": "Dummy",
    })
)

for nome_modelo, tabela_modelo in [
    ("Regressão Logística", tabela_logistica_resumo),
    ("KNN", tabela_knn_resumo),
    ("Árvore de Decisão", tabela_arvore_resumo),
]:
    comparacao_inicial_arvore = comparacao_inicial_arvore.merge(
        tabela_modelo[
            ["Métrica", "Média na validação"]
        ].rename(columns={
            "Média na validação": nome_modelo,
        }),
        on="Métrica",
        validate="one_to_one",
    )

comparacao_inicial_arvore[
    "Árvore - Dummy (pontos percentuais)"
] = (
    comparacao_inicial_arvore["Árvore de Decisão"]
    - comparacao_inicial_arvore["Dummy"]
) * 100

comparacao_inicial_arvore[
    "Árvore - Logística (pontos percentuais)"
] = (
    comparacao_inicial_arvore["Árvore de Decisão"]
    - comparacao_inicial_arvore["Regressão Logística"]
) * 100

print("\nResultados por divisão:")
print(
    tabela_arvore_divisoes.round(4).to_string(index=False)
)

print("\nResumo das métricas:")
print(
    tabela_arvore_resumo.round(4).to_string(index=False)
)

print("\nComparação das médias na validação:")
print(
    comparacao_inicial_arvore.round(4).to_string(index=False)
)

print("\nTempos médios:")
print(
    tabela_arvore_tempos.round(6).to_string(index=False)
)

arquivo_arvore_divisoes = (
    PASTA_RESULTADOS
    / "arvore_inicial_validacao_cruzada_divisoes.csv"
)

arquivo_arvore_resumo = (
    PASTA_RESULTADOS
    / "arvore_inicial_validacao_cruzada_resumo.csv"
)

arquivo_arvore_tempos = (
    PASTA_RESULTADOS
    / "arvore_inicial_validacao_cruzada_tempos.csv"
)

arquivo_comparacao_arvore = (
    PASTA_RESULTADOS
    / "comparacao_modelos_iniciais_ate_arvore.csv"
)

for tabela, caminho in [
    (tabela_arvore_divisoes, arquivo_arvore_divisoes),
    (tabela_arvore_resumo, arquivo_arvore_resumo),
    (tabela_arvore_tempos, arquivo_arvore_tempos),
    (comparacao_inicial_arvore, arquivo_comparacao_arvore),
]:
    tabela.to_csv(
        caminho,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        float_format="%.6f",
    )

    print("\nTabela salva em:", caminho)

print(
    "\nAs métricas estão na escala de 0 a 1. "
    "As diferenças estão em pontos percentuais."
)
print(
    "Uma árvore sem limite de profundidade pode apresentar "
    "desempenho muito alto no treino e inferior na validação."
)
print(
    "Profundidade, quantidade mínima por folha e outros "
    "hiperparâmetros serão avaliados posteriormente no treino."
)
print(
    "O tempo de ajuste inclui pré-processamento e classificador."
)
print(
    "O tempo de avaliação inclui transformação, previsões "
    "e cálculo das métricas."
)
print(
    "As árvores retornadas pertencem às cinco divisões. "
    "Nenhuma delas é o modelo final ajustado em todo o treino."
)
print(
    "pipeline_arvore permanece sem ajuste; "
    "o cross_validate treinou cópias independentes."
)

print("\nO conjunto de teste reservado não foi utilizado.")
print("\nBloco 14 concluído.")

# %% ==========================================================
# BLOCO 15 — Random Forest inicial
# =============================================================

from sklearn.ensemble import RandomForestClassifier

print("\n" + "=" * 70)
print("BLOCO 15 — RANDOM FOREST INICIAL")
print("=" * 70)

# Configuração inicial, sem busca de hiperparâmetros.
classificador_forest = RandomForestClassifier(
    n_estimators=200,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    class_weight=None,
    random_state=SEMENTE_ALEATORIA,
    n_jobs=1,
)

pipeline_forest = criar_pipeline(classificador_forest)

print("\nModelo: Random Forest")
print("Quantidade de árvores: 200")
print("Critério de divisão: Gini")
print("Profundidade máxima: sem limite")
print("Mínimo de registros por folha: 1")
print("Variáveis candidatas por divisão: raiz quadrada do total")
print("Amostragem com reposição por árvore: habilitada")
print("Pesos das classes: sem ponderação adicional")
print("Divisões da validação cruzada:", len(divisoes_cv))
print("Registros disponíveis:", len(X_treino))

print(
    "\nO pré-processamento será ajustado dentro de cada divisão."
)
print(
    "Random Forest não exige padronização, mas manteremos "
    "o Pipeline comum nesta comparação inicial."
)
print(
    "\nExecutando validação cruzada. Aguarde a conclusão..."
)

resultados_forest_cv = cross_validate(
    estimator=pipeline_forest,
    X=X_treino,
    y=y_treino,
    cv=divisoes_cv,
    scoring=metricas_cv,
    return_train_score=True,
    n_jobs=1,
    error_score="raise",
)

# Resultados individuais das cinco divisões.
linhas_forest_cv = []

for numero, (posicoes_treino, posicoes_validacao) in enumerate(
    divisoes_cv,
    start=1,
):
    indice_resultado = numero - 1

    linha = {
        "Divisão": numero,
        "Registros de treino": len(posicoes_treino),
        "Registros de validação": len(posicoes_validacao),
        "Tempo de ajuste (s)": resultados_forest_cv[
            "fit_time"
        ][indice_resultado],
        "Tempo de avaliação (s)": resultados_forest_cv[
            "score_time"
        ][indice_resultado],
    }

    for nome_metrica in metricas_cv:
        linha[f"Treino - {nome_metrica}"] = (
            resultados_forest_cv[f"train_{nome_metrica}"][
                indice_resultado
            ]
        )

        linha[f"Validação - {nome_metrica}"] = (
            resultados_forest_cv[f"test_{nome_metrica}"][
                indice_resultado
            ]
        )

    linhas_forest_cv.append(linha)

tabela_forest_divisoes = pd.DataFrame(linhas_forest_cv)

# Resumo das métricas entre as divisões.
linhas_resumo_forest = []

for nome_metrica in metricas_cv:
    valores_treino = resultados_forest_cv[
        f"train_{nome_metrica}"
    ]

    valores_validacao = resultados_forest_cv[
        f"test_{nome_metrica}"
    ]

    linhas_resumo_forest.append({
        "Modelo": "Random Forest — inicial",
        "Métrica": nome_metrica,
        "Média no treino": np.mean(valores_treino),
        "Desvio padrão no treino": np.std(
            valores_treino,
            ddof=1,
        ),
        "Média na validação": np.mean(valores_validacao),
        "Desvio padrão na validação": np.std(
            valores_validacao,
            ddof=1,
        ),
        "Diferença treino - validação": (
            np.mean(valores_treino)
            - np.mean(valores_validacao)
        ),
    })

tabela_forest_resumo = pd.DataFrame(linhas_resumo_forest)

tabela_forest_tempos = pd.DataFrame([{
    "Modelo": "Random Forest — inicial",
    "Divisões": len(divisoes_cv),
    "Tempo médio de ajuste (s)": np.mean(
        resultados_forest_cv["fit_time"]
    ),
    "Tempo médio de avaliação (s)": np.mean(
        resultados_forest_cv["score_time"]
    ),
}])

# Compara as médias de validação dos modelos já avaliados.
comparacao_inicial_forest = (
    tabela_dummy_resumo[
        ["Métrica", "Média na validação"]
    ]
    .rename(columns={
        "Média na validação": "Dummy",
    })
)

for nome_modelo, tabela_modelo in [
    ("Regressão Logística", tabela_logistica_resumo),
    ("KNN", tabela_knn_resumo),
    ("Árvore de Decisão", tabela_arvore_resumo),
    ("Random Forest", tabela_forest_resumo),
]:
    comparacao_inicial_forest = comparacao_inicial_forest.merge(
        tabela_modelo[
            ["Métrica", "Média na validação"]
        ].rename(columns={
            "Média na validação": nome_modelo,
        }),
        on="Métrica",
        validate="one_to_one",
    )

comparacao_inicial_forest[
    "Forest - Dummy (pontos percentuais)"
] = (
    comparacao_inicial_forest["Random Forest"]
    - comparacao_inicial_forest["Dummy"]
) * 100

comparacao_inicial_forest[
    "Forest - Logística (pontos percentuais)"
] = (
    comparacao_inicial_forest["Random Forest"]
    - comparacao_inicial_forest["Regressão Logística"]
) * 100

print("\nResultados por divisão:")
print(
    tabela_forest_divisoes.round(4).to_string(index=False)
)

print("\nResumo das métricas:")
print(
    tabela_forest_resumo.round(4).to_string(index=False)
)

print("\nComparação das médias na validação:")
print(
    comparacao_inicial_forest.round(4).to_string(index=False)
)

print("\nTempos médios:")
print(
    tabela_forest_tempos.round(6).to_string(index=False)
)

arquivo_forest_divisoes = (
    PASTA_RESULTADOS
    / "forest_inicial_validacao_cruzada_divisoes.csv"
)

arquivo_forest_resumo = (
    PASTA_RESULTADOS
    / "forest_inicial_validacao_cruzada_resumo.csv"
)

arquivo_forest_tempos = (
    PASTA_RESULTADOS
    / "forest_inicial_validacao_cruzada_tempos.csv"
)

arquivo_comparacao_forest = (
    PASTA_RESULTADOS
    / "comparacao_modelos_iniciais_ate_forest.csv"
)

for tabela, caminho in [
    (tabela_forest_divisoes, arquivo_forest_divisoes),
    (tabela_forest_resumo, arquivo_forest_resumo),
    (tabela_forest_tempos, arquivo_forest_tempos),
    (comparacao_inicial_forest, arquivo_comparacao_forest),
]:
    tabela.to_csv(
        caminho,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        float_format="%.6f",
    )

    print("\nTabela salva em:", caminho)

print(
    "\nAs métricas estão na escala de 0 a 1. "
    "As diferenças estão em pontos percentuais."
)
print(
    "Combinar árvores pode reduzir a variabilidade das previsões "
    "em relação a uma árvore isolada, mas não elimina "
    "automaticamente o sobreajuste."
)
print(
    "A diferença entre treino e validação será examinada "
    "junto com o desempenho nas divisões."
)
print(
    "O tempo de ajuste inclui pré-processamento e classificador."
)
print(
    "O tempo de avaliação inclui transformação, previsões "
    "e cálculo das métricas."
)
print(
    "Os hiperparâmetros serão ajustados posteriormente "
    "usando somente o conjunto de treino."
)
print(
    "pipeline_forest permanece sem ajuste; "
    "o cross_validate treinou cópias independentes."
)

print("\nO conjunto de teste reservado não foi utilizado.")
print("\nBloco 15 concluído.")

# %% ==========================================================
# BLOCO 16 — SVM inicial com kernels linear e RBF
# =============================================================

from sklearn.svm import SVC

print("\n" + "=" * 70)
print("BLOCO 16 — SVM INICIAL: LINEAR E RBF")
print("=" * 70)

# Configurações iniciais, sem busca de hiperparâmetros.
# C controla a penalização dos erros de classificação.
# gamma é utilizado pelo kernel RBF; não pelo linear.
# max_iter=-1 não impõe um limite fixo de iterações.
classificador_svm_linear = SVC(
    kernel="linear",
    C=1.0,
    tol=0.001,
    cache_size=200,
    class_weight=None,
    max_iter=-1,
    decision_function_shape="ovr",
    random_state=SEMENTE_ALEATORIA,
)

classificador_svm_rbf = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    tol=0.001,
    cache_size=200,
    class_weight=None,
    max_iter=-1,
    decision_function_shape="ovr",
    random_state=SEMENTE_ALEATORIA,
)

# Cada Pipeline recebe uma cópia independente do preprocessador.
pipeline_svm_linear = criar_pipeline(
    classificador_svm_linear
)

pipeline_svm_rbf = criar_pipeline(
    classificador_svm_rbf
)

print("\nModelos: SVM linear e SVM RBF")
print("C: 1.0 nos dois modelos")
print("Gamma do RBF: scale")
print("Pesos das classes: sem ponderação adicional")
print("Divisões da validação cruzada:", len(divisoes_cv))
print("Registros disponíveis:", len(X_treino))

print(
    "\nO pré-processamento será ajustado dentro de cada divisão."
)
print(
    "Serão usadas as mesmas divisões e métricas "
    "dos classificadores anteriores."
)
print(
    "Os dois kernels serão avaliados em sequência. "
    "Aguarde a conclusão."
)

# Guarda os resultados separadamente para uso posterior.
resultados_svm_cv = {}
tabelas_svm_divisoes = {}
tabelas_svm_resumo = {}
tabelas_svm_tempos = {}

configuracoes_svm = [
    ("linear", "SVM linear — inicial", pipeline_svm_linear),
    ("rbf", "SVM RBF — inicial", pipeline_svm_rbf),
]

for identificador, nome_modelo, pipeline_modelo in configuracoes_svm:
    print("\n" + "-" * 70)
    print("AVALIANDO:", nome_modelo)
    print("-" * 70)

    # Mantém o tratamento de convergência usado na Logística.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)

        resultados = cross_validate(
            estimator=pipeline_modelo,
            X=X_treino,
            y=y_treino,
            cv=divisoes_cv,
            scoring=metricas_cv,
            return_train_score=True,
            n_jobs=1,
            error_score="raise",
        )

    resultados_svm_cv[identificador] = resultados

    # Uma linha para cada divisão da validação cruzada.
    linhas_divisoes_svm = []

    for numero, (
        posicoes_treino,
        posicoes_validacao,
    ) in enumerate(divisoes_cv, start=1):
        indice_resultado = numero - 1

        linha = {
            "Divisão": numero,
            "Registros de treino": len(posicoes_treino),
            "Registros de validação": len(posicoes_validacao),
            "Tempo de ajuste (s)": resultados[
                "fit_time"
            ][indice_resultado],
            "Tempo de avaliação (s)": resultados[
                "score_time"
            ][indice_resultado],
        }

        for nome_metrica in metricas_cv:
            linha[f"Treino - {nome_metrica}"] = (
                resultados[f"train_{nome_metrica}"][
                    indice_resultado
                ]
            )

            linha[f"Validação - {nome_metrica}"] = (
                resultados[f"test_{nome_metrica}"][
                    indice_resultado
                ]
            )

        linhas_divisoes_svm.append(linha)

    tabela_divisoes_svm = pd.DataFrame(linhas_divisoes_svm)

    # Resumo das métricas entre as cinco divisões.
    linhas_resumo_svm = []

    for nome_metrica in metricas_cv:
        valores_treino = resultados[
            f"train_{nome_metrica}"
        ]

        valores_validacao = resultados[
            f"test_{nome_metrica}"
        ]

        linhas_resumo_svm.append({
            "Modelo": nome_modelo,
            "Métrica": nome_metrica,
            "Média no treino": np.mean(valores_treino),
            "Desvio padrão no treino": np.std(
                valores_treino,
                ddof=1,
            ),
            "Média na validação": np.mean(valores_validacao),
            "Desvio padrão na validação": np.std(
                valores_validacao,
                ddof=1,
            ),
            "Diferença treino - validação": (
                np.mean(valores_treino)
                - np.mean(valores_validacao)
            ),
        })

    tabela_resumo_svm = pd.DataFrame(linhas_resumo_svm)

    tabela_tempos_svm = pd.DataFrame([{
        "Modelo": nome_modelo,
        "Divisões": len(divisoes_cv),
        "Tempo médio de ajuste (s)": np.mean(
            resultados["fit_time"]
        ),
        "Tempo médio de avaliação (s)": np.mean(
            resultados["score_time"]
        ),
    }])

    tabelas_svm_divisoes[identificador] = tabela_divisoes_svm
    tabelas_svm_resumo[identificador] = tabela_resumo_svm
    tabelas_svm_tempos[identificador] = tabela_tempos_svm

    print("\nResultados por divisão:")
    print(
        tabela_divisoes_svm.round(4).to_string(index=False)
    )

    print("\nResumo das métricas:")
    print(
        tabela_resumo_svm.round(4).to_string(index=False)
    )

    print("\nTempos médios:")
    print(
        tabela_tempos_svm.round(6).to_string(index=False)
    )

    prefixo_svm = f"svm_{identificador}_inicial"

    for tabela, sufixo in [
        (tabela_divisoes_svm, "divisoes"),
        (tabela_resumo_svm, "resumo"),
        (tabela_tempos_svm, "tempos"),
    ]:
        caminho_svm = (
            PASTA_RESULTADOS
            / f"{prefixo_svm}_validacao_cruzada_{sufixo}.csv"
        )

        tabela.to_csv(
            caminho_svm,
            index=False,
            sep=";",
            encoding="utf-8-sig",
            decimal=",",
            float_format="%.6f",
        )

        print("\nTabela salva em:", caminho_svm)

# Nomes explícitos, seguindo o padrão dos blocos anteriores.
resultados_svm_linear_cv = resultados_svm_cv["linear"]
resultados_svm_rbf_cv = resultados_svm_cv["rbf"]

tabela_svm_linear_divisoes = tabelas_svm_divisoes["linear"]
tabela_svm_rbf_divisoes = tabelas_svm_divisoes["rbf"]

tabela_svm_linear_resumo = tabelas_svm_resumo["linear"]
tabela_svm_rbf_resumo = tabelas_svm_resumo["rbf"]

tabela_svm_linear_tempos = tabelas_svm_tempos["linear"]
tabela_svm_rbf_tempos = tabelas_svm_tempos["rbf"]

# Compara as médias de validação de todos os modelos já avaliados.
comparacao_inicial_svm = (
    tabela_dummy_resumo[
        ["Métrica", "Média na validação"]
    ]
    .rename(columns={
        "Média na validação": "Dummy",
    })
)

for nome_modelo, tabela_modelo in [
    ("Regressão Logística", tabela_logistica_resumo),
    ("KNN", tabela_knn_resumo),
    ("Árvore de Decisão", tabela_arvore_resumo),
    ("Random Forest", tabela_forest_resumo),
    ("SVM linear", tabela_svm_linear_resumo),
    ("SVM RBF", tabela_svm_rbf_resumo),
]:
    comparacao_inicial_svm = comparacao_inicial_svm.merge(
        tabela_modelo[
            ["Métrica", "Média na validação"]
        ].rename(columns={
            "Média na validação": nome_modelo,
        }),
        on="Métrica",
        validate="one_to_one",
    )

print("\n" + "=" * 70)
print("COMPARAÇÃO DOS MODELOS INICIAIS ATÉ SVM")
print("=" * 70)

print(
    comparacao_inicial_svm.round(4).to_string(index=False)
)

arquivo_comparacao_svm = (
    PASTA_RESULTADOS
    / "comparacao_modelos_iniciais_ate_svm.csv"
)

comparacao_inicial_svm.to_csv(
    arquivo_comparacao_svm,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    decimal=",",
    float_format="%.6f",
)

print("\nComparação salva em:", arquivo_comparacao_svm)

print(
    "\nAs métricas estão na escala de 0 a 1."
)
print(
    "A média macro atribui o mesmo peso às três classes."
)
print(
    "A diferença entre treino e validação deve ser interpretada "
    "junto com o desempenho e a variação entre as divisões."
)
print(
    "O tempo de ajuste inclui pré-processamento e classificador."
)
print(
    "O tempo de avaliação inclui transformação, previsões "
    "e cálculo das métricas."
)
print(
    "Estas são configurações iniciais. O ajuste de "
    "hiperparâmetros será realizado posteriormente no treino."
)
print(
    "As médias, isoladamente, não demonstram "
    "significância estatística entre os modelos."
)
print(
    "pipeline_svm_linear e pipeline_svm_rbf permanecem "
    "sem ajuste; o cross_validate treinou cópias independentes."
)

print("\nO conjunto de teste reservado não foi utilizado.")
print("\nBloco 16 concluído.")

# %% ==========================================================
# BLOCO 17 — Naive Bayes inicial
# =============================================================

from sklearn.naive_bayes import GaussianNB

print("\n" + "=" * 70)
print("BLOCO 17 — NAIVE BAYES INICIAL")
print("=" * 70)

# Configuração inicial, sem busca de hiperparâmetros.
# priors=None: aprende as proporções das classes no treino.
# var_smoothing: acrescenta uma parcela à variância para
# melhorar a estabilidade numérica dos cálculos.
classificador_nb = GaussianNB(
    priors=None,
    var_smoothing=1e-9,
)

pipeline_nb = criar_pipeline(classificador_nb)

print("\nModelo: Gaussian Naive Bayes")
print("Probabilidades iniciais das classes: aprendidas no treino")
print("Var smoothing: 1e-9")
print("Divisões da validação cruzada:", len(divisoes_cv))
print("Registros disponíveis:", len(X_treino))

print(
    "\nO pré-processamento será ajustado dentro de cada divisão."
)
print(
    "Serão usadas as mesmas divisões e métricas "
    "dos classificadores anteriores."
)
print(
    "O Pipeline existente produz uma matriz densa, "
    "compatível com GaussianNB."
)
print(
    "A hipótese gaussiana é uma aproximação, especialmente "
    "para indicadores binários e categorias codificadas."
)
print(
    "\nExecutando validação cruzada. Aguarde a conclusão..."
)

resultados_nb_cv = cross_validate(
    estimator=pipeline_nb,
    X=X_treino,
    y=y_treino,
    cv=divisoes_cv,
    scoring=metricas_cv,
    return_train_score=True,
    n_jobs=1,
    error_score="raise",
)

# Uma linha para cada divisão da validação cruzada.
linhas_nb_cv = []

for numero, (posicoes_treino, posicoes_validacao) in enumerate(
    divisoes_cv,
    start=1,
):
    indice_resultado = numero - 1

    linha = {
        "Divisão": numero,
        "Registros de treino": len(posicoes_treino),
        "Registros de validação": len(posicoes_validacao),
        "Tempo de ajuste (s)": resultados_nb_cv[
            "fit_time"
        ][indice_resultado],
        "Tempo de avaliação (s)": resultados_nb_cv[
            "score_time"
        ][indice_resultado],
    }

    for nome_metrica in metricas_cv:
        linha[f"Treino - {nome_metrica}"] = (
            resultados_nb_cv[f"train_{nome_metrica}"][
                indice_resultado
            ]
        )

        linha[f"Validação - {nome_metrica}"] = (
            resultados_nb_cv[f"test_{nome_metrica}"][
                indice_resultado
            ]
        )

    linhas_nb_cv.append(linha)

tabela_nb_divisoes = pd.DataFrame(linhas_nb_cv)

# Resumo das métricas entre as cinco divisões.
linhas_resumo_nb = []

for nome_metrica in metricas_cv:
    valores_treino = resultados_nb_cv[
        f"train_{nome_metrica}"
    ]

    valores_validacao = resultados_nb_cv[
        f"test_{nome_metrica}"
    ]

    linhas_resumo_nb.append({
        "Modelo": "Gaussian Naive Bayes — inicial",
        "Métrica": nome_metrica,
        "Média no treino": np.mean(valores_treino),
        "Desvio padrão no treino": np.std(
            valores_treino,
            ddof=1,
        ),
        "Média na validação": np.mean(valores_validacao),
        "Desvio padrão na validação": np.std(
            valores_validacao,
            ddof=1,
        ),
        "Diferença treino - validação": (
            np.mean(valores_treino)
            - np.mean(valores_validacao)
        ),
    })

tabela_nb_resumo = pd.DataFrame(linhas_resumo_nb)

tabela_nb_tempos = pd.DataFrame([{
    "Modelo": "Gaussian Naive Bayes — inicial",
    "Divisões": len(divisoes_cv),
    "Tempo médio de ajuste (s)": np.mean(
        resultados_nb_cv["fit_time"]
    ),
    "Tempo médio de avaliação (s)": np.mean(
        resultados_nb_cv["score_time"]
    ),
}])

# Comparação das médias de validação de todos os modelos iniciais.
comparacao_inicial_nb = (
    tabela_dummy_resumo[
        ["Métrica", "Média na validação"]
    ]
    .rename(columns={
        "Média na validação": "Dummy",
    })
)

for nome_modelo, tabela_modelo in [
    ("Regressão Logística", tabela_logistica_resumo),
    ("KNN", tabela_knn_resumo),
    ("Árvore de Decisão", tabela_arvore_resumo),
    ("Random Forest", tabela_forest_resumo),
    ("SVM linear", tabela_svm_linear_resumo),
    ("SVM RBF", tabela_svm_rbf_resumo),
    ("Gaussian Naive Bayes", tabela_nb_resumo),
]:
    comparacao_inicial_nb = comparacao_inicial_nb.merge(
        tabela_modelo[
            ["Métrica", "Média na validação"]
        ].rename(columns={
            "Média na validação": nome_modelo,
        }),
        on="Métrica",
        validate="one_to_one",
    )

print("\nResultados por divisão:")
print(
    tabela_nb_divisoes.round(4).to_string(index=False)
)

print("\nResumo das métricas:")
print(
    tabela_nb_resumo.round(4).to_string(index=False)
)

print("\nTempos médios:")
print(
    tabela_nb_tempos.round(6).to_string(index=False)
)

print("\n" + "=" * 70)
print("COMPARAÇÃO DOS MODELOS INICIAIS ATÉ NAIVE BAYES")
print("=" * 70)

print(
    comparacao_inicial_nb.round(4).to_string(index=False)
)

arquivo_nb_divisoes = (
    PASTA_RESULTADOS
    / "nb_inicial_validacao_cruzada_divisoes.csv"
)

arquivo_nb_resumo = (
    PASTA_RESULTADOS
    / "nb_inicial_validacao_cruzada_resumo.csv"
)

arquivo_nb_tempos = (
    PASTA_RESULTADOS
    / "nb_inicial_validacao_cruzada_tempos.csv"
)

arquivo_comparacao_nb = (
    PASTA_RESULTADOS
    / "comparacao_modelos_iniciais_ate_nb.csv"
)

for tabela, caminho in [
    (tabela_nb_divisoes, arquivo_nb_divisoes),
    (tabela_nb_resumo, arquivo_nb_resumo),
    (tabela_nb_tempos, arquivo_nb_tempos),
    (comparacao_inicial_nb, arquivo_comparacao_nb),
]:
    tabela.to_csv(
        caminho,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        float_format="%.6f",
    )

    print("\nTabela salva em:", caminho)

print(
    "\nAs métricas estão na escala de 0 a 1."
)
print(
    "A média macro atribui o mesmo peso às três classes."
)
print(
    "Naive Bayes pressupõe independência condicional "
    "das entradas dada a classe."
)
print(
    "As correlações observadas e as dependências entre "
    "categorias codificadas limitam essa hipótese."
)
print(
    "Categorias raras podem produzir variâncias muito pequenas. "
    "O var_smoothing será avaliado no ajuste de hiperparâmetros."
)
print(
    "Um resultado baixo nesta configuração não permite "
    "concluir que toda a família Naive Bayes é inadequada."
)
print(
    "O tempo de ajuste inclui pré-processamento e classificador."
)
print(
    "O tempo de avaliação inclui transformação, previsões "
    "e cálculo das métricas."
)
print(
    "Esta configuração é inicial. Ainda não foi escolhido "
    "um modelo vencedor."
)
print(
    "pipeline_nb permanece sem ajuste; "
    "o cross_validate treinou cópias independentes."
)

print("\nO conjunto de teste reservado não foi utilizado.")
print("\nBloco 17 concluído.")

# %% ==========================================================
# BLOCO 18 — Consolidação da comparação inicial
# =============================================================

print("\n" + "=" * 70)
print("BLOCO 18 — CONSOLIDAÇÃO DA COMPARAÇÃO INICIAL")
print("=" * 70)

METRICA_PRINCIPAL = "f1_macro"

print("\nMétrica principal para orientar os ajustes:", METRICA_PRINCIPAL)
print("Justificativa: atribuir o mesmo peso às três classes.")
print("As demais métricas também serão consideradas.")
print("Nenhum modelo será treinado novamente neste bloco.")

# Tabelas produzidas nos blocos anteriores.
modelos_para_consolidar = [
    (
        "Dummy",
        tabela_dummy_resumo,
        tabela_dummy_tempos,
    ),
    (
        "Regressão Logística",
        tabela_logistica_resumo,
        tabela_logistica_tempos,
    ),
    (
        "KNN",
        tabela_knn_resumo,
        tabela_knn_tempos,
    ),
    (
        "Árvore de Decisão",
        tabela_arvore_resumo,
        tabela_arvore_tempos,
    ),
    (
        "Random Forest",
        tabela_forest_resumo,
        tabela_forest_tempos,
    ),
    (
        "SVM linear",
        tabela_svm_linear_resumo,
        tabela_svm_linear_tempos,
    ),
    (
        "SVM RBF",
        tabela_svm_rbf_resumo,
        tabela_svm_rbf_tempos,
    ),
    (
        "Gaussian Naive Bayes",
        tabela_nb_resumo,
        tabela_nb_tempos,
    ),
]

linhas_comparacao_inicial = []

for nome_modelo, tabela_resumo, tabela_tempos in modelos_para_consolidar:
    if tabela_resumo["Métrica"].duplicated().any():
        raise ValueError(
            f"Existem métricas repetidas no resumo de {nome_modelo}."
        )

    resumo_por_metrica = tabela_resumo.set_index("Métrica")

    metricas_faltantes = (
        set(metricas_cv) - set(resumo_por_metrica.index)
    )

    if metricas_faltantes:
        raise ValueError(
            f"Métricas ausentes em {nome_modelo}: "
            f"{sorted(metricas_faltantes)}"
        )

    if len(tabela_tempos) != 1:
        raise ValueError(
            f"A tabela de tempos de {nome_modelo} "
            "precisa conter exatamente uma linha."
        )

    if int(tabela_tempos.iloc[0]["Divisões"]) != len(divisoes_cv):
        raise ValueError(
            f"A quantidade de divisões de {nome_modelo} "
            "difere da configuração atual."
        )

    linha_f1 = resumo_por_metrica.loc[METRICA_PRINCIPAL]
    linha_tempo = tabela_tempos.iloc[0]

    linha_comparacao = {
        "Modelo": nome_modelo,
        "Acurácia na validação": resumo_por_metrica.loc[
            "acuracia", "Média na validação"
        ],
        "Precisão macro na validação": resumo_por_metrica.loc[
            "precisao_macro", "Média na validação"
        ],
        "Recall macro na validação": resumo_por_metrica.loc[
            "recall_macro", "Média na validação"
        ],
        "F1 macro na validação": linha_f1["Média na validação"],
        "DP do F1 macro na validação": linha_f1[
            "Desvio padrão na validação"
        ],
        "F1 macro no treino": linha_f1["Média no treino"],
        "Diferença F1 treino - validação (pp)": (
            linha_f1["Média no treino"]
            - linha_f1["Média na validação"]
        ) * 100,
        "Tempo médio de ajuste (s)": linha_tempo[
            "Tempo médio de ajuste (s)"
        ],
        "Tempo médio de avaliação (s)": linha_tempo[
            "Tempo médio de avaliação (s)"
        ],
    }

    linhas_comparacao_inicial.append(linha_comparacao)

comparacao_inicial_consolidada = pd.DataFrame(
    linhas_comparacao_inicial
)

# Confere se as métricas e os tempos são números finitos.
colunas_numericas_comparacao = (
    comparacao_inicial_consolidada.columns.drop("Modelo")
)

if not np.isfinite(
    comparacao_inicial_consolidada[
        colunas_numericas_comparacao
    ].to_numpy(dtype=float)
).all():
    raise ValueError(
        "A comparação contém valores ausentes ou não finitos."
    )

# Ordenação descritiva pelas médias, sem declarar vencedor final.
comparacao_inicial_consolidada = (
    comparacao_inicial_consolidada
    .sort_values(
        by="F1 macro na validação",
        ascending=False,
        kind="stable",
    )
    .reset_index(drop=True)
)

f1_dummy_inicial = comparacao_inicial_consolidada.loc[
    comparacao_inicial_consolidada["Modelo"].eq("Dummy"),
    "F1 macro na validação",
].iloc[0]

comparacao_inicial_consolidada[
    "Diferença F1 para Dummy (pp)"
] = (
    comparacao_inicial_consolidada["F1 macro na validação"]
    - f1_dummy_inicial
) * 100

print("\nComparação ordenada pelo F1 macro médio na validação:")
print(
    comparacao_inicial_consolidada.round(4).to_string(
        index=False
    )
)

arquivo_comparacao_consolidada = (
    PASTA_RESULTADOS
    / "comparacao_modelos_iniciais_consolidada.csv"
)

comparacao_inicial_consolidada.to_csv(
    arquivo_comparacao_consolidada,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    decimal=",",
    float_format="%.6f",
)

print("\nTabela salva em:", arquivo_comparacao_consolidada)

# Gráfico: maior F1 macro aparece no topo.
dados_grafico_comparacao = (
    comparacao_inicial_consolidada
    .sort_values(
        by="F1 macro na validação",
        ascending=True,
        kind="stable",
    )
    .reset_index(drop=True)
)

posicoes_modelos = np.arange(len(dados_grafico_comparacao))

f1_validacao_percentual = (
    dados_grafico_comparacao["F1 macro na validação"]
    .to_numpy() * 100
)

dp_validacao_percentual = (
    dados_grafico_comparacao["DP do F1 macro na validação"]
    .to_numpy() * 100
)

f1_treino_percentual = (
    dados_grafico_comparacao["F1 macro no treino"]
    .to_numpy() * 100
)

cores_comparacao = [
    "#999999" if modelo == "Dummy" else "#4472C4"
    for modelo in dados_grafico_comparacao["Modelo"]
]

fig_comparacao_inicial = plt.figure(
    figsize=(12, 8),
    layout="constrained",
)

fig_comparacao_inicial.canvas.manager.set_window_title(
    "Figura 4 — Comparação inicial dos classificadores"
)

grade_comparacao = fig_comparacao_inicial.add_gridspec(
    nrows=2,
    ncols=1,
    height_ratios=[1, 0.17],
)

ax_comparacao = fig_comparacao_inicial.add_subplot(
    grade_comparacao[0, 0]
)

ax_comparacao.barh(
    posicoes_modelos,
    f1_validacao_percentual,
    xerr=dp_validacao_percentual,
    color=cores_comparacao,
    height=0.6,
    alpha=0.9,
    capsize=4,
    error_kw={
        "ecolor": "#222222",
        "elinewidth": 1.2,
    },
    label="Validação: média ± 1 desvio padrão",
)

ax_comparacao.scatter(
    f1_treino_percentual,
    posicoes_modelos,
    color="#C44E52",
    marker="D",
    s=45,
    zorder=3,
    label="Treino: média",
)

for posicao, media, desvio in zip(
    posicoes_modelos,
    f1_validacao_percentual,
    dp_validacao_percentual,
):
    ax_comparacao.text(
        media + desvio + 1.0,
        posicao,
        f"{media:.2f}%".replace(".", ","),
        va="center",
        fontsize=9,
    )

ax_comparacao.set_yticks(posicoes_modelos)
ax_comparacao.set_yticklabels(
    dados_grafico_comparacao["Modelo"]
)

ax_comparacao.set_xlim(0, 105)
ax_comparacao.set_xticks(np.arange(0, 101, 10))
ax_comparacao.set_xlabel("F1 macro (%)")
ax_comparacao.set_title(
    "Comparação inicial dos classificadores\n"
    "Validação cruzada estratificada com cinco divisões",
    fontsize=14,
    pad=14,
)

ax_comparacao.grid(axis="x", alpha=0.25)
ax_comparacao.set_axisbelow(True)
ax_comparacao.spines["top"].set_visible(False)
ax_comparacao.spines["right"].set_visible(False)
ax_comparacao.legend(
    loc="lower right",
    fontsize=8,
)

# Rodapé em área própria para evitar sobreposição.
ax_rodape_comparacao = fig_comparacao_inicial.add_subplot(
    grade_comparacao[1, 0]
)
ax_rodape_comparacao.set_axis_off()

ax_rodape_comparacao.text(
    0.5,
    0.5,
    "Fonte: resultados dos Blocos 11 a 17 — elaboração do grupo.\n"
    "Mesmas divisões do treino para todos os modelos. "
    "Teste reservado não utilizado.\n"
    "Barras de erro mostram a variação entre divisões; "
    "não são intervalos de confiança.",
    transform=ax_rodape_comparacao.transAxes,
    ha="center",
    va="center",
    fontsize=9,
)

arquivo_figura_comparacao = (
    PASTA_RESULTADOS
    / "figura_04_comparacao_modelos_iniciais.png"
)

fig_comparacao_inicial.savefig(
    arquivo_figura_comparacao,
    dpi=300,
    bbox_inches="tight",
)

print("\nFigura salva em:", arquivo_figura_comparacao)

print(
    "\nAs métricas da tabela estão na escala de 0 a 1."
)
print(
    "No gráfico, o F1 macro está expresso em percentual."
)
print(
    "As diferenças identificadas por 'pp' "
    "estão em pontos percentuais."
)
print(
    "Os tempos correspondem às execuções desta sessão "
    "e podem variar entre execuções."
)
print(
    "Uma diferença pequena entre treino e validação "
    "não garante bom desempenho: considere também o F1 obtido."
)
print(
    "O F1 no treino do KNN inclui a possibilidade "
    "de cada registro ser seu próprio vizinho."
)
print(
    "A ordenação das médias não comprova "
    "superioridade estatística entre os modelos."
)
print(
    "Esta comparação reúne configurações iniciais. "
    "O próximo estágio será o ajuste de hiperparâmetros "
    "usando somente os dados de treino."
)

print("\nNenhum modelo foi treinado novamente neste bloco.")
print("O conjunto de teste reservado não foi utilizado.")
print("\nFeche a Figura 4 para concluir o Bloco 18.")

plt.show()
plt.close(fig_comparacao_inicial)

print("\nBloco 18 concluído.")

# %% ==========================================================
# BLOCO 19 — Ajuste da Regressão Logística
# =============================================================

from time import perf_counter

from sklearn.model_selection import GridSearchCV, ParameterGrid

print("\n" + "=" * 70)
print("BLOCO 19 — AJUSTE DA REGRESSÃO LOGÍSTICA")
print("=" * 70)

# Mantém o solver e o limite de iterações do modelo inicial.
# A busca altera somente C e class_weight.
pipeline_logistica_busca = criar_pipeline(
    LogisticRegression(
        solver="lbfgs",
        max_iter=3000,
        random_state=SEMENTE_ALEATORIA,
    )
)

grade_logistica = {
    "modelo__C": [
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
        1000.0,
    ],
    "modelo__class_weight": [
        None,
        "balanced",
    ],
}

numero_configuracoes_logistica = len(
    list(ParameterGrid(grade_logistica))
)

print("\nModelo: Regressão Logística")
print("Solver: lbfgs")
print("Limite de iterações por ajuste: 3000")
print("Valores de C:", grade_logistica["modelo__C"])
print("Pesos das classes: None e balanced")
print("Métrica de seleção:", METRICA_PRINCIPAL)
print("Configurações:", numero_configuracoes_logistica)
print("Divisões por configuração:", len(divisoes_cv))
print(
    "Ajustes na validação cruzada:",
    numero_configuracoes_logistica * len(divisoes_cv),
)
print("Registros disponíveis:", len(X_treino))

print(
    "\nO pré-processamento será aprendido dentro "
    "de cada divisão da validação cruzada."
)
print(
    "A configuração inicial C=1.0 e class_weight=None "
    "também está incluída na busca."
)
print(
    "Após a seleção, o melhor Pipeline desta grade "
    "será ajustado em todo o conjunto de treino."
)
print(
    "\nIniciando busca. Aguarde; ela executa vários ajustes..."
)

busca_logistica = GridSearchCV(
    estimator=pipeline_logistica_busca,
    param_grid=grade_logistica,
    scoring=metricas_cv,
    refit=METRICA_PRINCIPAL,
    cv=divisoes_cv,
    return_train_score=True,
    n_jobs=1,
    verbose=1,
    error_score="raise",
)

inicio_busca_logistica = perf_counter()

# Interrompe a execução se algum ajuste não convergir.
with warnings.catch_warnings():
    warnings.simplefilter("error", ConvergenceWarning)

    busca_logistica.fit(
        X_treino,
        y_treino,
    )

tempo_total_busca_logistica = (
    perf_counter() - inicio_busca_logistica
)

# Este Pipeline já foi ajustado nos 3.096 registros de treino.
# Ainda não é a escolha final entre todas as famílias.
melhor_pipeline_logistica = busca_logistica.best_estimator_

indice_melhor_logistica = busca_logistica.best_index_
resultados_grade_logistica = busca_logistica.cv_results_

print("\nBusca concluída.")
print("Parâmetros selecionados:", busca_logistica.best_params_)
print(
    "F1 macro médio na validação:",
    f"{busca_logistica.best_score_:.4f}",
)
print(
    "Tempo total da busca, incluindo ajuste final:",
    f"{tempo_total_busca_logistica:.2f} s",
)
print(
    "Tempo do ajuste final em todo o treino:",
    f"{busca_logistica.refit_time_:.2f} s",
)

# Preserva todos os resultados retornados pelo GridSearchCV.
# As chaves test_* representam validação interna do treino.
tabela_logistica_busca_completa = pd.DataFrame(
    resultados_grade_logistica
)

# Tabela legível com uma linha por configuração.
linhas_grade_logistica = []

for indice, parametros in enumerate(
    resultados_grade_logistica["params"]
):
    linha = {
        "C": parametros["modelo__C"],
        "Peso das classes": (
            "Sem ponderação"
            if parametros["modelo__class_weight"] is None
            else "balanced"
        ),
        "Posição por F1 macro": int(
            resultados_grade_logistica[
                "rank_test_f1_macro"
            ][indice]
        ),
        "Selecionada": indice == indice_melhor_logistica,
        "Tempo médio de ajuste (s)": (
            resultados_grade_logistica["mean_fit_time"][indice]
        ),
        "Tempo médio de avaliação (s)": (
            resultados_grade_logistica["mean_score_time"][indice]
        ),
    }

    for nome_metrica in metricas_cv:
        valores_treino = np.array([
            resultados_grade_logistica[
                f"split{divisao}_train_{nome_metrica}"
            ][indice]
            for divisao in range(len(divisoes_cv))
        ])

        valores_validacao = np.array([
            resultados_grade_logistica[
                f"split{divisao}_test_{nome_metrica}"
            ][indice]
            for divisao in range(len(divisoes_cv))
        ])

        linha[f"Treino - {nome_metrica}"] = np.mean(
            valores_treino
        )
        linha[f"Validação - {nome_metrica}"] = np.mean(
            valores_validacao
        )
        linha[f"DP validação - {nome_metrica}"] = np.std(
            valores_validacao,
            ddof=1,
        )

    linha["Diferença F1 treino - validação (pp)"] = (
        linha["Treino - f1_macro"]
        - linha["Validação - f1_macro"]
    ) * 100

    linhas_grade_logistica.append(linha)

tabela_logistica_grade = (
    pd.DataFrame(linhas_grade_logistica)
    .sort_values(
        by="Posição por F1 macro",
        kind="stable",
    )
    .reset_index(drop=True)
)

# Resumo da configuração selecionada.
# Mantém o padrão dos blocos anteriores e usa ddof=1.
linhas_logistica_ajustada_resumo = []

for nome_metrica in metricas_cv:
    valores_treino = np.array([
        resultados_grade_logistica[
            f"split{divisao}_train_{nome_metrica}"
        ][indice_melhor_logistica]
        for divisao in range(len(divisoes_cv))
    ])

    valores_validacao = np.array([
        resultados_grade_logistica[
            f"split{divisao}_test_{nome_metrica}"
        ][indice_melhor_logistica]
        for divisao in range(len(divisoes_cv))
    ])

    linhas_logistica_ajustada_resumo.append({
        "Modelo": "Regressão Logística — ajustada",
        "Métrica": nome_metrica,
        "Média no treino": np.mean(valores_treino),
        "Desvio padrão no treino": np.std(
            valores_treino,
            ddof=1,
        ),
        "Média na validação": np.mean(valores_validacao),
        "Desvio padrão na validação": np.std(
            valores_validacao,
            ddof=1,
        ),
        "Diferença treino - validação": (
            np.mean(valores_treino)
            - np.mean(valores_validacao)
        ),
    })

tabela_logistica_ajustada_resumo = pd.DataFrame(
    linhas_logistica_ajustada_resumo
)

# Comparação descritiva com a configuração inicial.
comparacao_logistica_ajuste = (
    tabela_logistica_resumo[
        ["Métrica", "Média na validação"]
    ]
    .rename(columns={
        "Média na validação": "Inicial - validação",
    })
    .merge(
        tabela_logistica_ajustada_resumo[
            ["Métrica", "Média na validação"]
        ].rename(columns={
            "Média na validação": "Selecionada - validação",
        }),
        on="Métrica",
        validate="one_to_one",
    )
)

comparacao_logistica_ajuste[
    "Diferença selecionada - inicial (pp)"
] = (
    comparacao_logistica_ajuste["Selecionada - validação"]
    - comparacao_logistica_ajuste["Inicial - validação"]
) * 100

# Documenta os parâmetros e os diferentes tempos.
tabela_logistica_ajustada_tempos = pd.DataFrame([{
    "Modelo": "Regressão Logística — ajustada",
    "C selecionado": busca_logistica.best_params_["modelo__C"],
    "Peso das classes": (
        "Sem ponderação"
        if busca_logistica.best_params_[
            "modelo__class_weight"
        ] is None
        else "balanced"
    ),
    "Métrica de seleção": METRICA_PRINCIPAL,
    "Configurações": numero_configuracoes_logistica,
    "Divisões": len(divisoes_cv),
    "Tempo médio de ajuste (s)": resultados_grade_logistica[
        "mean_fit_time"
    ][indice_melhor_logistica],
    "Tempo médio de avaliação (s)": resultados_grade_logistica[
        "mean_score_time"
    ][indice_melhor_logistica],
    "Tempo do ajuste final no treino (s)": (
        busca_logistica.refit_time_
    ),
    "Tempo total da busca com ajuste final (s)": (
        tempo_total_busca_logistica
    ),
}])

print("\nConfigurações ordenadas pelo F1 macro:")
print(
    tabela_logistica_grade.round(4).to_string(index=False)
)

print("\nResumo da configuração selecionada:")
print(
    tabela_logistica_ajustada_resumo.round(4).to_string(
        index=False
    )
)

print("\nComparação com a configuração inicial:")
print(
    comparacao_logistica_ajuste.round(4).to_string(
        index=False
    )
)

print("\nParâmetros e tempos:")
print(
    tabela_logistica_ajustada_tempos.round(6).to_string(
        index=False
    )
)

# Salva as tabelas sem sobrescrever os resultados iniciais.
for tabela, nome_arquivo in [
    (
        tabela_logistica_busca_completa,
        "logistica_ajuste_busca_completa.csv",
    ),
    (
        tabela_logistica_grade,
        "logistica_ajuste_configuracoes.csv",
    ),
    (
        tabela_logistica_ajustada_resumo,
        "logistica_ajustada_validacao_cruzada_resumo.csv",
    ),
    (
        tabela_logistica_ajustada_tempos,
        "logistica_ajustada_tempos_parametros.csv",
    ),
    (
        comparacao_logistica_ajuste,
        "comparacao_logistica_inicial_ajustada.csv",
    ),
]:
    caminho = PASTA_RESULTADOS / nome_arquivo

    tabela.to_csv(
        caminho,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        float_format="%.6f",
    )

    print("\nTabela salva em:", caminho)

print(
    "\nOs desvios padrão das tabelas resumidas usam ddof=1."
)
print(
    "O arquivo busca_completa preserva o cv_results_ original, "
    "cujos campos std_* usam a convenção do scikit-learn."
)
print(
    "A melhor média foi selecionada nas mesmas divisões "
    "usadas para comparar os parâmetros. Ela pode ser otimista "
    "e não constitui uma avaliação independente."
)
print(
    "A diferença em relação à configuração inicial "
    "é descritiva; não comprova superioridade estatística."
)
print(
    "melhor_pipeline_logistica está ajustado em todo o treino "
    "e disponível na memória desta execução."
)
print(
    "O modelo ainda não foi salvo em arquivo nem escolhido "
    "como vencedor entre todas as famílias."
)

print("\nO conjunto de teste reservado não foi utilizado.")
print("\nBloco 19 concluído.")

# %% ==========================================================
# BLOCO 20 — Ajuste de hiperparâmetros do KNN
# =============================================================

print("\n" + "=" * 70)
print("BLOCO 20 — AJUSTE DE HIPERPARÂMETROS DO KNN")
print("=" * 70)

# Mantém o mesmo pré-processamento e método de busca
# de vizinhos usados na configuração inicial.
pipeline_knn_busca = criar_pipeline(
    KNeighborsClassifier(
        metric="minkowski",
        algorithm="brute",
        n_jobs=1,
    )
)

grade_knn = {
    "modelo__n_neighbors": [3, 5, 9, 15, 25, 41],
    "modelo__weights": ["uniform", "distance"],
    "modelo__p": [1, 2],
}

numero_configuracoes_knn = len(
    list(ParameterGrid(grade_knn))
)

# Confere se todos os valores de k cabem nas partes de treino.
menor_treino_cv_knn = min(
    len(posicoes_treino)
    for posicoes_treino, _ in divisoes_cv
)

if max(grade_knn["modelo__n_neighbors"]) > menor_treino_cv_knn:
    raise ValueError(
        "Existe um número de vizinhos maior que a quantidade "
        "de registros de treino disponível em uma divisão."
    )

print("\nModelo: KNN")
print("Vizinhos:", grade_knn["modelo__n_neighbors"])
print("Pesos:", grade_knn["modelo__weights"])
print("Distâncias: Manhattan (p=1) e Euclidiana (p=2)")
print("Métrica de seleção:", METRICA_PRINCIPAL)
print("Configurações:", numero_configuracoes_knn)
print("Divisões por configuração:", len(divisoes_cv))
print(
    "Ajustes na validação cruzada:",
    numero_configuracoes_knn * len(divisoes_cv),
)
print("Registros disponíveis:", len(X_treino))

print(
    "\nA configuração inicial k=5, weights=uniform e p=2 "
    "está incluída na busca."
)
print(
    "O pré-processamento será aprendido dentro "
    "de cada divisão da validação cruzada."
)
print(
    "Após a seleção, o melhor Pipeline desta grade "
    "será ajustado em todo o conjunto de treino."
)
print("\nIniciando busca. Aguarde a conclusão...")

busca_knn = GridSearchCV(
    estimator=pipeline_knn_busca,
    param_grid=grade_knn,
    scoring=metricas_cv,
    refit=METRICA_PRINCIPAL,
    cv=divisoes_cv,
    return_train_score=True,
    n_jobs=1,
    verbose=1,
    error_score="raise",
)

inicio_busca_knn = perf_counter()

busca_knn.fit(
    X_treino,
    y_treino,
)

tempo_total_busca_knn = (
    perf_counter() - inicio_busca_knn
)

# Pipeline ajustado em todo o treino, sem avaliação no teste.
melhor_pipeline_knn = busca_knn.best_estimator_

indice_melhor_knn = busca_knn.best_index_
resultados_grade_knn = busca_knn.cv_results_

print("\nBusca concluída.")
print("Parâmetros selecionados:", busca_knn.best_params_)
print(
    "F1 macro médio na validação:",
    f"{busca_knn.best_score_:.4f}",
)
print(
    "Tempo total da busca, incluindo ajuste final:",
    f"{tempo_total_busca_knn:.2f} s",
)
print(
    "Tempo do ajuste final em todo o treino:",
    f"{busca_knn.refit_time_:.2f} s",
)

# Preserva todos os resultados originais da busca.
tabela_knn_busca_completa = pd.DataFrame(
    resultados_grade_knn
)

# Uma linha por configuração.
linhas_grade_knn = []

for indice, parametros in enumerate(
    resultados_grade_knn["params"]
):
    linha = {
        "Vizinhos": parametros["modelo__n_neighbors"],
        "Pesos": parametros["modelo__weights"],
        "p": parametros["modelo__p"],
        "Distância": (
            "Manhattan"
            if parametros["modelo__p"] == 1
            else "Euclidiana"
        ),
        "Posição por F1 macro": int(
            resultados_grade_knn["rank_test_f1_macro"][indice]
        ),
        "Selecionada": indice == indice_melhor_knn,
        "Tempo médio de ajuste (s)": resultados_grade_knn[
            "mean_fit_time"
        ][indice],
        "Tempo médio de avaliação (s)": resultados_grade_knn[
            "mean_score_time"
        ][indice],
    }

    for nome_metrica in metricas_cv:
        valores_treino = np.array([
            resultados_grade_knn[
                f"split{divisao}_train_{nome_metrica}"
            ][indice]
            for divisao in range(len(divisoes_cv))
        ])

        valores_validacao = np.array([
            resultados_grade_knn[
                f"split{divisao}_test_{nome_metrica}"
            ][indice]
            for divisao in range(len(divisoes_cv))
        ])

        linha[f"Treino - {nome_metrica}"] = np.mean(
            valores_treino
        )
        linha[f"Validação - {nome_metrica}"] = np.mean(
            valores_validacao
        )
        linha[f"DP validação - {nome_metrica}"] = np.std(
            valores_validacao,
            ddof=1,
        )

    linha["Diferença F1 treino - validação (pp)"] = (
        linha["Treino - f1_macro"]
        - linha["Validação - f1_macro"]
    ) * 100

    linhas_grade_knn.append(linha)

tabela_knn_grade = (
    pd.DataFrame(linhas_grade_knn)
    .sort_values(
        by="Posição por F1 macro",
        kind="stable",
    )
    .reset_index(drop=True)
)

# Resumo da configuração selecionada.
linhas_knn_ajustado_resumo = []

for nome_metrica in metricas_cv:
    valores_treino = np.array([
        resultados_grade_knn[
            f"split{divisao}_train_{nome_metrica}"
        ][indice_melhor_knn]
        for divisao in range(len(divisoes_cv))
    ])

    valores_validacao = np.array([
        resultados_grade_knn[
            f"split{divisao}_test_{nome_metrica}"
        ][indice_melhor_knn]
        for divisao in range(len(divisoes_cv))
    ])

    linhas_knn_ajustado_resumo.append({
        "Modelo": "KNN — ajustado",
        "Métrica": nome_metrica,
        "Média no treino": np.mean(valores_treino),
        "Desvio padrão no treino": np.std(
            valores_treino,
            ddof=1,
        ),
        "Média na validação": np.mean(valores_validacao),
        "Desvio padrão na validação": np.std(
            valores_validacao,
            ddof=1,
        ),
        "Diferença treino - validação": (
            np.mean(valores_treino)
            - np.mean(valores_validacao)
        ),
    })

tabela_knn_ajustado_resumo = pd.DataFrame(
    linhas_knn_ajustado_resumo
)

# Comparação descritiva com o KNN inicial.
comparacao_knn_ajuste = (
    tabela_knn_resumo[
        ["Métrica", "Média na validação"]
    ]
    .rename(columns={
        "Média na validação": "Inicial - validação",
    })
    .merge(
        tabela_knn_ajustado_resumo[
            ["Métrica", "Média na validação"]
        ].rename(columns={
            "Média na validação": "Selecionada - validação",
        }),
        on="Métrica",
        validate="one_to_one",
    )
)

comparacao_knn_ajuste[
    "Diferença selecionada - inicial (pp)"
] = (
    comparacao_knn_ajuste["Selecionada - validação"]
    - comparacao_knn_ajuste["Inicial - validação"]
) * 100

tabela_knn_ajustado_tempos = pd.DataFrame([{
    "Modelo": "KNN — ajustado",
    "Vizinhos selecionados": busca_knn.best_params_[
        "modelo__n_neighbors"
    ],
    "Pesos selecionados": busca_knn.best_params_[
        "modelo__weights"
    ],
    "p selecionado": busca_knn.best_params_["modelo__p"],
    "Métrica de seleção": METRICA_PRINCIPAL,
    "Configurações": numero_configuracoes_knn,
    "Divisões": len(divisoes_cv),
    "Tempo médio de ajuste (s)": resultados_grade_knn[
        "mean_fit_time"
    ][indice_melhor_knn],
    "Tempo médio de avaliação (s)": resultados_grade_knn[
        "mean_score_time"
    ][indice_melhor_knn],
    "Tempo do ajuste final no treino (s)": (
        busca_knn.refit_time_
    ),
    "Tempo total da busca com ajuste final (s)": (
        tempo_total_busca_knn
    ),
}])

print("\nConfigurações ordenadas pelo F1 macro:")
print(
    tabela_knn_grade.round(4).to_string(index=False)
)

print("\nResumo da configuração selecionada:")
print(
    tabela_knn_ajustado_resumo.round(4).to_string(index=False)
)

print("\nComparação com a configuração inicial:")
print(
    comparacao_knn_ajuste.round(4).to_string(index=False)
)

print("\nParâmetros e tempos:")
print(
    tabela_knn_ajustado_tempos.round(6).to_string(index=False)
)

# Preserva os arquivos dos experimentos anteriores.
for tabela, nome_arquivo in [
    (
        tabela_knn_busca_completa,
        "knn_ajuste_busca_completa.csv",
    ),
    (
        tabela_knn_grade,
        "knn_ajuste_configuracoes.csv",
    ),
    (
        tabela_knn_ajustado_resumo,
        "knn_ajustado_validacao_cruzada_resumo.csv",
    ),
    (
        tabela_knn_ajustado_tempos,
        "knn_ajustado_tempos_parametros.csv",
    ),
    (
        comparacao_knn_ajuste,
        "comparacao_knn_inicial_ajustado.csv",
    ),
]:
    caminho = PASTA_RESULTADOS / nome_arquivo

    tabela.to_csv(
        caminho,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        float_format="%.6f",
    )

    print("\nTabela salva em:", caminho)

print(
    "\nOs desvios padrão das tabelas resumidas usam ddof=1."
)
print(
    "O arquivo busca_completa preserva o cv_results_ original, "
    "cujos campos std_* usam a convenção do scikit-learn."
)
print(
    "Na avaliação do treino, cada registro pode ser "
    "seu próprio vizinho. Com pesos por distância, isso "
    "pode produzir pontuações muito altas no treino."
)
print(
    "Por isso, a diferença treino - validação do KNN "
    "deve ser interpretada considerando essa particularidade."
)
print(
    "A seleção é feita pelo F1 macro na validação, "
    "e não pela pontuação no treino."
)
print(
    "A melhor média foi selecionada nas mesmas divisões "
    "usadas para comparar os parâmetros. Ela pode ser otimista "
    "e não constitui uma avaliação independente."
)
print(
    "A diferença em relação à configuração inicial "
    "é descritiva; não comprova superioridade estatística."
)
print(
    "melhor_pipeline_knn está ajustado em todo o treino "
    "e disponível na memória desta execução."
)
print(
    "O modelo ainda não foi salvo em arquivo nem escolhido "
    "como vencedor entre todas as famílias."
)

print("\nO conjunto de teste reservado não foi utilizado.")
print("\nBloco 20 concluído.")

# %% ==========================================================
# BLOCO 21 — Ajuste da Árvore de Decisão
# =============================================================

print("\n" + "=" * 70)
print("BLOCO 21 — AJUSTE DA ÁRVORE DE DECISÃO")
print("=" * 70)

# Mantém o critério Gini e a semente do modelo inicial.
# A busca avalia controles de complexidade e pesos das classes.
pipeline_arvore_busca = criar_pipeline(
    DecisionTreeClassifier(
        criterion="gini",
        min_samples_split=2,
        random_state=SEMENTE_ALEATORIA,
    )
)

grade_arvore = {
    "modelo__max_depth": [3, 5, 10, None],
    "modelo__min_samples_leaf": [1, 5, 15, 30],
    "modelo__class_weight": [None, "balanced"],
}

numero_configuracoes_arvore = len(
    list(ParameterGrid(grade_arvore))
)

print("\nModelo: Árvore de Decisão")
print("Critério: Gini")
print("Profundidades:", grade_arvore["modelo__max_depth"])
print(
    "Mínimo de registros por folha:",
    grade_arvore["modelo__min_samples_leaf"],
)
print("Pesos das classes: None e balanced")
print("Métrica de seleção:", METRICA_PRINCIPAL)
print("Configurações:", numero_configuracoes_arvore)
print("Divisões por configuração:", len(divisoes_cv))
print(
    "Ajustes na validação cruzada:",
    numero_configuracoes_arvore * len(divisoes_cv),
)
print("Registros disponíveis:", len(X_treino))

print(
    "\nA configuração inicial, sem limite de profundidade, "
    "com uma amostra mínima por folha e sem ponderação, "
    "está incluída na busca."
)
print(
    "O pré-processamento será aprendido dentro "
    "de cada divisão da validação cruzada."
)
print(
    "Manteremos o Pipeline comum dos experimentos anteriores."
)
print(
    "Após a seleção, o melhor Pipeline desta grade "
    "será ajustado em todo o conjunto de treino."
)
print("\nIniciando busca. Aguarde a conclusão...")

busca_arvore = GridSearchCV(
    estimator=pipeline_arvore_busca,
    param_grid=grade_arvore,
    scoring=metricas_cv,
    refit=METRICA_PRINCIPAL,
    cv=divisoes_cv,
    return_train_score=True,
    n_jobs=1,
    verbose=1,
    error_score="raise",
)

inicio_busca_arvore = perf_counter()

busca_arvore.fit(
    X_treino,
    y_treino,
)

tempo_total_busca_arvore = (
    perf_counter() - inicio_busca_arvore
)

# Pipeline selecionado e ajustado em todo o treino.
melhor_pipeline_arvore = busca_arvore.best_estimator_

indice_melhor_arvore = busca_arvore.best_index_
resultados_grade_arvore = busca_arvore.cv_results_

arvore_ajustada_treino = (
    melhor_pipeline_arvore.named_steps["modelo"]
)

print("\nBusca concluída.")
print("Parâmetros selecionados:", busca_arvore.best_params_)
print(
    "F1 macro médio na validação:",
    f"{busca_arvore.best_score_:.4f}",
)
print(
    "Profundidade da árvore ajustada em todo o treino:",
    arvore_ajustada_treino.get_depth(),
)
print(
    "Folhas da árvore ajustada em todo o treino:",
    arvore_ajustada_treino.get_n_leaves(),
)
print(
    "Tempo total da busca, incluindo ajuste final:",
    f"{tempo_total_busca_arvore:.2f} s",
)
print(
    "Tempo do ajuste final em todo o treino:",
    f"{busca_arvore.refit_time_:.2f} s",
)

# Preserva todos os resultados originais da busca.
tabela_arvore_busca_completa = pd.DataFrame(
    resultados_grade_arvore
)

# Uma linha por configuração.
linhas_grade_arvore = []

for indice, parametros in enumerate(
    resultados_grade_arvore["params"]
):
    profundidade = parametros["modelo__max_depth"]

    linha = {
        "Profundidade máxima": (
            "Sem limite"
            if profundidade is None
            else str(profundidade)
        ),
        "Mínimo por folha": parametros[
            "modelo__min_samples_leaf"
        ],
        "Peso das classes": (
            "Sem ponderação"
            if parametros["modelo__class_weight"] is None
            else "balanced"
        ),
        "Posição por F1 macro": int(
            resultados_grade_arvore[
                "rank_test_f1_macro"
            ][indice]
        ),
        "Selecionada": indice == indice_melhor_arvore,
        "Tempo médio de ajuste (s)": resultados_grade_arvore[
            "mean_fit_time"
        ][indice],
        "Tempo médio de avaliação (s)": resultados_grade_arvore[
            "mean_score_time"
        ][indice],
    }

    for nome_metrica in metricas_cv:
        valores_treino = np.array([
            resultados_grade_arvore[
                f"split{divisao}_train_{nome_metrica}"
            ][indice]
            for divisao in range(len(divisoes_cv))
        ])

        valores_validacao = np.array([
            resultados_grade_arvore[
                f"split{divisao}_test_{nome_metrica}"
            ][indice]
            for divisao in range(len(divisoes_cv))
        ])

        linha[f"Treino - {nome_metrica}"] = np.mean(
            valores_treino
        )
        linha[f"Validação - {nome_metrica}"] = np.mean(
            valores_validacao
        )
        linha[f"DP validação - {nome_metrica}"] = np.std(
            valores_validacao,
            ddof=1,
        )

    linha["Diferença F1 treino - validação (pp)"] = (
        linha["Treino - f1_macro"]
        - linha["Validação - f1_macro"]
    ) * 100

    linhas_grade_arvore.append(linha)

tabela_arvore_grade = (
    pd.DataFrame(linhas_grade_arvore)
    .sort_values(
        by="Posição por F1 macro",
        kind="stable",
    )
    .reset_index(drop=True)
)

# Resumo da configuração selecionada.
linhas_arvore_ajustada_resumo = []

for nome_metrica in metricas_cv:
    valores_treino = np.array([
        resultados_grade_arvore[
            f"split{divisao}_train_{nome_metrica}"
        ][indice_melhor_arvore]
        for divisao in range(len(divisoes_cv))
    ])

    valores_validacao = np.array([
        resultados_grade_arvore[
            f"split{divisao}_test_{nome_metrica}"
        ][indice_melhor_arvore]
        for divisao in range(len(divisoes_cv))
    ])

    linhas_arvore_ajustada_resumo.append({
        "Modelo": "Árvore de Decisão — ajustada",
        "Métrica": nome_metrica,
        "Média no treino": np.mean(valores_treino),
        "Desvio padrão no treino": np.std(
            valores_treino,
            ddof=1,
        ),
        "Média na validação": np.mean(valores_validacao),
        "Desvio padrão na validação": np.std(
            valores_validacao,
            ddof=1,
        ),
        "Diferença treino - validação": (
            np.mean(valores_treino)
            - np.mean(valores_validacao)
        ),
    })

tabela_arvore_ajustada_resumo = pd.DataFrame(
    linhas_arvore_ajustada_resumo
)

# Comparação descritiva com a árvore inicial.
comparacao_arvore_ajuste = (
    tabela_arvore_resumo[
        ["Métrica", "Média na validação"]
    ]
    .rename(columns={
        "Média na validação": "Inicial - validação",
    })
    .merge(
        tabela_arvore_ajustada_resumo[
            ["Métrica", "Média na validação"]
        ].rename(columns={
            "Média na validação": "Selecionada - validação",
        }),
        on="Métrica",
        validate="one_to_one",
    )
)

comparacao_arvore_ajuste[
    "Diferença selecionada - inicial (pp)"
] = (
    comparacao_arvore_ajuste["Selecionada - validação"]
    - comparacao_arvore_ajuste["Inicial - validação"]
) * 100

profundidade_selecionada_arvore = (
    busca_arvore.best_params_["modelo__max_depth"]
)

tabela_arvore_ajustada_tempos = pd.DataFrame([{
    "Modelo": "Árvore de Decisão — ajustada",
    "Profundidade máxima selecionada": (
        "Sem limite"
        if profundidade_selecionada_arvore is None
        else str(profundidade_selecionada_arvore)
    ),
    "Mínimo por folha selecionado": busca_arvore.best_params_[
        "modelo__min_samples_leaf"
    ],
    "Peso das classes": (
        "Sem ponderação"
        if busca_arvore.best_params_[
            "modelo__class_weight"
        ] is None
        else "balanced"
    ),
    "Profundidade no ajuste final": (
        arvore_ajustada_treino.get_depth()
    ),
    "Folhas no ajuste final": (
        arvore_ajustada_treino.get_n_leaves()
    ),
    "Métrica de seleção": METRICA_PRINCIPAL,
    "Configurações": numero_configuracoes_arvore,
    "Divisões": len(divisoes_cv),
    "Tempo médio de ajuste (s)": resultados_grade_arvore[
        "mean_fit_time"
    ][indice_melhor_arvore],
    "Tempo médio de avaliação (s)": resultados_grade_arvore[
        "mean_score_time"
    ][indice_melhor_arvore],
    "Tempo do ajuste final no treino (s)": (
        busca_arvore.refit_time_
    ),
    "Tempo total da busca com ajuste final (s)": (
        tempo_total_busca_arvore
    ),
}])

print("\nConfigurações ordenadas pelo F1 macro:")
print(
    tabela_arvore_grade.round(4).to_string(index=False)
)

print("\nResumo da configuração selecionada:")
print(
    tabela_arvore_ajustada_resumo.round(4).to_string(
        index=False
    )
)

print("\nComparação com a configuração inicial:")
print(
    comparacao_arvore_ajuste.round(4).to_string(index=False)
)

print("\nParâmetros, tamanho da árvore e tempos:")
print(
    tabela_arvore_ajustada_tempos.round(6).to_string(
        index=False
    )
)

# Preserva os arquivos dos experimentos anteriores.
for tabela, nome_arquivo in [
    (
        tabela_arvore_busca_completa,
        "arvore_ajuste_busca_completa.csv",
    ),
    (
        tabela_arvore_grade,
        "arvore_ajuste_configuracoes.csv",
    ),
    (
        tabela_arvore_ajustada_resumo,
        "arvore_ajustada_validacao_cruzada_resumo.csv",
    ),
    (
        tabela_arvore_ajustada_tempos,
        "arvore_ajustada_tempos_parametros.csv",
    ),
    (
        comparacao_arvore_ajuste,
        "comparacao_arvore_inicial_ajustada.csv",
    ),
]:
    caminho = PASTA_RESULTADOS / nome_arquivo

    tabela.to_csv(
        caminho,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        float_format="%.6f",
    )

    print("\nTabela salva em:", caminho)

print(
    "\nOs desvios padrão das tabelas resumidas usam ddof=1."
)
print(
    "O arquivo busca_completa preserva o cv_results_ original, "
    "cujos campos std_* usam a convenção do scikit-learn."
)
print(
    "Limitar profundidade e aumentar o mínimo por folha "
    "restringem a complexidade da árvore."
)
print(
    "A redução da diferença entre treino e validação "
    "deve ser examinada junto com o F1 macro obtido."
)
print(
    "A profundidade e as folhas exibidas pertencem à árvore "
    "ajustada em todo o treino, não às árvores das divisões."
)
print(
    "A melhor média foi selecionada nas mesmas divisões "
    "usadas para comparar os parâmetros. Ela pode ser otimista "
    "e não constitui uma avaliação independente."
)
print(
    "A diferença em relação à configuração inicial "
    "é descritiva; não comprova superioridade estatística."
)
print(
    "melhor_pipeline_arvore está ajustado em todo o treino "
    "e disponível na memória desta execução."
)
print(
    "O modelo ainda não foi salvo em arquivo nem escolhido "
    "como vencedor entre todas as famílias."
)

print("\nO conjunto de teste reservado não foi utilizado.")
print("\nBloco 21 concluído.")

# %% ==========================================================
# BLOCO 22 — Ajuste do Random Forest
# =============================================================

print("\n" + "=" * 70)
print("BLOCO 22 — AJUSTE DO RANDOM FOREST")
print("=" * 70)

# Mantém as 200 árvores e os demais parâmetros do modelo inicial.
# A busca avalia complexidade das árvores e pesos das classes.
pipeline_forest_busca = criar_pipeline(
    RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        min_samples_split=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=SEMENTE_ALEATORIA,
        n_jobs=1,
    )
)

grade_forest = {
    "modelo__max_depth": [5, 10, None],
    "modelo__min_samples_leaf": [1, 5, 15],
    "modelo__class_weight": [None, "balanced"],
}

numero_configuracoes_forest = len(
    list(ParameterGrid(grade_forest))
)

print("\nModelo: Random Forest")
print("Quantidade de árvores por ajuste: 200")
print("Profundidades:", grade_forest["modelo__max_depth"])
print(
    "Mínimo de registros por folha:",
    grade_forest["modelo__min_samples_leaf"],
)
print("Pesos das classes: None e balanced")
print("Métrica de seleção:", METRICA_PRINCIPAL)
print("Configurações:", numero_configuracoes_forest)
print("Divisões por configuração:", len(divisoes_cv))
print(
    "Ajustes na validação cruzada:",
    numero_configuracoes_forest * len(divisoes_cv),
)
print("Registros disponíveis:", len(X_treino))

print(
    "\nA configuração inicial está incluída na busca."
)
print(
    "O pré-processamento será aprendido dentro "
    "de cada divisão da validação cruzada."
)
print(
    "Após a seleção, o melhor Pipeline desta grade "
    "será ajustado em todo o conjunto de treino."
)
print(
    "\nIniciando busca. Cada ajuste treina 200 árvores; "
    "a execução pode levar alguns minutos."
)

busca_forest = GridSearchCV(
    estimator=pipeline_forest_busca,
    param_grid=grade_forest,
    scoring=metricas_cv,
    refit=METRICA_PRINCIPAL,
    cv=divisoes_cv,
    return_train_score=True,
    n_jobs=1,
    verbose=2,
    error_score="raise",
)

inicio_busca_forest = perf_counter()

busca_forest.fit(
    X_treino,
    y_treino,
)

tempo_total_busca_forest = (
    perf_counter() - inicio_busca_forest
)

# Pipeline selecionado e ajustado em todo o treino.
melhor_pipeline_forest = busca_forest.best_estimator_

indice_melhor_forest = busca_forest.best_index_
resultados_grade_forest = busca_forest.cv_results_

print("\nBusca concluída.")
print("Parâmetros selecionados:", busca_forest.best_params_)
print(
    "F1 macro médio na validação:",
    f"{busca_forest.best_score_:.4f}",
)
print(
    "Tempo total da busca, incluindo ajuste final:",
    f"{tempo_total_busca_forest:.2f} s",
)
print(
    "Tempo do ajuste final em todo o treino:",
    f"{busca_forest.refit_time_:.2f} s",
)

# Preserva todos os resultados originais da busca.
tabela_forest_busca_completa = pd.DataFrame(
    resultados_grade_forest
)

# Uma linha por configuração.
linhas_grade_forest = []

for indice, parametros in enumerate(
    resultados_grade_forest["params"]
):
    profundidade = parametros["modelo__max_depth"]

    linha = {
        "Profundidade máxima": (
            "Sem limite"
            if profundidade is None
            else str(profundidade)
        ),
        "Mínimo por folha": parametros[
            "modelo__min_samples_leaf"
        ],
        "Peso das classes": (
            "Sem ponderação"
            if parametros["modelo__class_weight"] is None
            else "balanced"
        ),
        "Posição por F1 macro": int(
            resultados_grade_forest[
                "rank_test_f1_macro"
            ][indice]
        ),
        "Selecionada": indice == indice_melhor_forest,
        "Tempo médio de ajuste (s)": resultados_grade_forest[
            "mean_fit_time"
        ][indice],
        "Tempo médio de avaliação (s)": resultados_grade_forest[
            "mean_score_time"
        ][indice],
    }

    for nome_metrica in metricas_cv:
        valores_treino = np.array([
            resultados_grade_forest[
                f"split{divisao}_train_{nome_metrica}"
            ][indice]
            for divisao in range(len(divisoes_cv))
        ])

        valores_validacao = np.array([
            resultados_grade_forest[
                f"split{divisao}_test_{nome_metrica}"
            ][indice]
            for divisao in range(len(divisoes_cv))
        ])

        linha[f"Treino - {nome_metrica}"] = np.mean(
            valores_treino
        )
        linha[f"Validação - {nome_metrica}"] = np.mean(
            valores_validacao
        )
        linha[f"DP validação - {nome_metrica}"] = np.std(
            valores_validacao,
            ddof=1,
        )

    linha["Diferença F1 treino - validação (pp)"] = (
        linha["Treino - f1_macro"]
        - linha["Validação - f1_macro"]
    ) * 100

    linhas_grade_forest.append(linha)

tabela_forest_grade = (
    pd.DataFrame(linhas_grade_forest)
    .sort_values(
        by="Posição por F1 macro",
        kind="stable",
    )
    .reset_index(drop=True)
)

# Resumo da configuração selecionada.
linhas_forest_ajustada_resumo = []

for nome_metrica in metricas_cv:
    valores_treino = np.array([
        resultados_grade_forest[
            f"split{divisao}_train_{nome_metrica}"
        ][indice_melhor_forest]
        for divisao in range(len(divisoes_cv))
    ])

    valores_validacao = np.array([
        resultados_grade_forest[
            f"split{divisao}_test_{nome_metrica}"
        ][indice_melhor_forest]
        for divisao in range(len(divisoes_cv))
    ])

    linhas_forest_ajustada_resumo.append({
        "Modelo": "Random Forest — ajustada",
        "Métrica": nome_metrica,
        "Média no treino": np.mean(valores_treino),
        "Desvio padrão no treino": np.std(
            valores_treino,
            ddof=1,
        ),
        "Média na validação": np.mean(valores_validacao),
        "Desvio padrão na validação": np.std(
            valores_validacao,
            ddof=1,
        ),
        "Diferença treino - validação": (
            np.mean(valores_treino)
            - np.mean(valores_validacao)
        ),
    })

tabela_forest_ajustada_resumo = pd.DataFrame(
    linhas_forest_ajustada_resumo
)

# Comparação descritiva com o Random Forest inicial.
comparacao_forest_ajuste = (
    tabela_forest_resumo[
        ["Métrica", "Média na validação"]
    ]
    .rename(columns={
        "Média na validação": "Inicial - validação",
    })
    .merge(
        tabela_forest_ajustada_resumo[
            ["Métrica", "Média na validação"]
        ].rename(columns={
            "Média na validação": "Selecionada - validação",
        }),
        on="Métrica",
        validate="one_to_one",
    )
)

comparacao_forest_ajuste[
    "Diferença selecionada - inicial (pp)"
] = (
    comparacao_forest_ajuste["Selecionada - validação"]
    - comparacao_forest_ajuste["Inicial - validação"]
) * 100

profundidade_selecionada_forest = (
    busca_forest.best_params_["modelo__max_depth"]
)

tabela_forest_ajustada_tempos = pd.DataFrame([{
    "Modelo": "Random Forest — ajustada",
    "Quantidade de árvores": 200,
    "Profundidade máxima selecionada": (
        "Sem limite"
        if profundidade_selecionada_forest is None
        else str(profundidade_selecionada_forest)
    ),
    "Mínimo por folha selecionado": busca_forest.best_params_[
        "modelo__min_samples_leaf"
    ],
    "Peso das classes": (
        "Sem ponderação"
        if busca_forest.best_params_[
            "modelo__class_weight"
        ] is None
        else "balanced"
    ),
    "Métrica de seleção": METRICA_PRINCIPAL,
    "Configurações": numero_configuracoes_forest,
    "Divisões": len(divisoes_cv),
    "Tempo médio de ajuste (s)": resultados_grade_forest[
        "mean_fit_time"
    ][indice_melhor_forest],
    "Tempo médio de avaliação (s)": resultados_grade_forest[
        "mean_score_time"
    ][indice_melhor_forest],
    "Tempo do ajuste final no treino (s)": (
        busca_forest.refit_time_
    ),
    "Tempo total da busca com ajuste final (s)": (
        tempo_total_busca_forest
    ),
}])

print("\nConfigurações ordenadas pelo F1 macro:")
print(
    tabela_forest_grade.round(4).to_string(index=False)
)

print("\nResumo da configuração selecionada:")
print(
    tabela_forest_ajustada_resumo.round(4).to_string(
        index=False
    )
)

print("\nComparação com a configuração inicial:")
print(
    comparacao_forest_ajuste.round(4).to_string(index=False)
)

print("\nParâmetros e tempos:")
print(
    tabela_forest_ajustada_tempos.round(6).to_string(
        index=False
    )
)

# Preserva os arquivos dos experimentos anteriores.
for tabela, nome_arquivo in [
    (
        tabela_forest_busca_completa,
        "forest_ajuste_busca_completa.csv",
    ),
    (
        tabela_forest_grade,
        "forest_ajuste_configuracoes.csv",
    ),
    (
        tabela_forest_ajustada_resumo,
        "forest_ajustada_validacao_cruzada_resumo.csv",
    ),
    (
        tabela_forest_ajustada_tempos,
        "forest_ajustada_tempos_parametros.csv",
    ),
    (
        comparacao_forest_ajuste,
        "comparacao_forest_inicial_ajustada.csv",
    ),
]:
    caminho = PASTA_RESULTADOS / nome_arquivo

    tabela.to_csv(
        caminho,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        float_format="%.6f",
    )

    print("\nTabela salva em:", caminho)

print(
    "\nOs desvios padrão das tabelas resumidas usam ddof=1."
)
print(
    "O arquivo busca_completa preserva o cv_results_ original, "
    "cujos campos std_* usam a convenção do scikit-learn."
)
print(
    "Esta busca mantém 200 árvores e max_features=sqrt. "
    "Ela não explora todos os hiperparâmetros possíveis."
)
print(
    "A redução da diferença entre treino e validação "
    "deve ser examinada junto com o F1 macro obtido."
)
print(
    "A melhor média foi selecionada nas mesmas divisões "
    "usadas para comparar os parâmetros. Ela pode ser otimista "
    "e não constitui uma avaliação independente."
)
print(
    "A diferença em relação à configuração inicial "
    "é descritiva; não comprova superioridade estatística."
)
print(
    "melhor_pipeline_forest está ajustado em todo o treino "
    "e disponível na memória desta execução."
)
print(
    "O modelo ainda não foi salvo em arquivo nem escolhido "
    "como vencedor entre todas as famílias."
)

print("\nO conjunto de teste reservado não foi utilizado.")
print("\nBloco 22 concluído.")

# %% ==========================================================
# BLOCO 23 — Ajuste de SVM linear e RBF
# =============================================================

import warnings
from time import perf_counter

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, ParameterGrid

print("\n" + "=" * 70)
print("BLOCO 23 — AJUSTE DE SVM LINEAR E RBF")
print("=" * 70)

# Mantém os classificadores iniciais do Bloco 16.
# clone cria estimadores independentes e sem ajuste.
# criar_pipeline inclui uma cópia do preprocessador.
#
# C controla a penalização dos erros.
# gamma controla o alcance da influência no kernel RBF.
# class_weight="balanced" calcula pesos a partir das
# frequências das classes em cada parte de treino.

grade_svm_linear = {
    "modelo__C": [0.01, 0.1, 1.0, 10.0],
    "modelo__class_weight": [None, "balanced"],
}

grade_svm_rbf = {
    "modelo__C": [0.1, 1.0, 10.0],
    "modelo__gamma": ["scale", 0.001, 0.01],
    "modelo__class_weight": [None, "balanced"],
}

numero_configuracoes_svm_linear = len(
    list(ParameterGrid(grade_svm_linear))
)

numero_configuracoes_svm_rbf = len(
    list(ParameterGrid(grade_svm_rbf))
)

numero_configuracoes_svm_total = (
    numero_configuracoes_svm_linear
    + numero_configuracoes_svm_rbf
)

print("\nMétrica de seleção:", METRICA_PRINCIPAL)
print("Divisões por configuração:", len(divisoes_cv))
print("Registros disponíveis no treino:", len(X_treino))
print(
    "Configurações do SVM linear:",
    numero_configuracoes_svm_linear,
)
print(
    "Configurações do SVM RBF:",
    numero_configuracoes_svm_rbf,
)
print(
    "Total de configurações:",
    numero_configuracoes_svm_total,
)
print(
    "Total de ajustes na validação cruzada:",
    numero_configuracoes_svm_total * len(divisoes_cv),
)

print(
    "\nAs configurações iniciais dos dois kernels "
    "estão incluídas nas respectivas grades."
)
print(
    "O pré-processamento será ajustado dentro "
    "de cada divisão da validação cruzada."
)
print(
    "Cada busca selecionará uma configuração pelo F1 macro "
    "e fará um ajuste final em todo o treino."
)
print(
    "As duas buscas serão executadas em sequência. "
    "O tempo dependerá do computador e das configurações."
)

# Dicionários preservam os resultados dos dois kernels.
buscas_svm = {}
melhores_pipelines_svm = {}
tabelas_svm_busca_completa = {}
tabelas_svm_grade_ajuste = {}
tabelas_svm_ajustadas_resumo = {}
tabelas_svm_ajustadas_tempos = {}
comparacoes_svm_ajuste = {}

configuracoes_svm_ajuste = [
    (
        "linear",
        "SVM linear — ajustado",
        classificador_svm_linear,
        grade_svm_linear,
        tabela_svm_linear_resumo,
    ),
    (
        "rbf",
        "SVM RBF — ajustado",
        classificador_svm_rbf,
        grade_svm_rbf,
        tabela_svm_rbf_resumo,
    ),
]

for (
    identificador,
    nome_modelo,
    classificador_inicial,
    grade,
    tabela_inicial,
) in configuracoes_svm_ajuste:

    print("\n" + "-" * 70)
    print("AJUSTANDO:", nome_modelo)
    print("-" * 70)

    numero_configuracoes = len(list(ParameterGrid(grade)))

    print("\nGrade:", grade)
    print("Configurações:", numero_configuracoes)
    print(
        "Ajustes na validação cruzada:",
        numero_configuracoes * len(divisoes_cv),
    )

    pipeline_svm_busca = criar_pipeline(
        clone(classificador_inicial)
    )

    busca_svm = GridSearchCV(
        estimator=pipeline_svm_busca,
        param_grid=grade,
        scoring=metricas_cv,
        refit=METRICA_PRINCIPAL,
        cv=divisoes_cv,
        return_train_score=True,
        n_jobs=1,
        verbose=2,
        error_score="raise",
    )

    inicio_busca_svm = perf_counter()

    # Mantém o tratamento de convergência do Bloco 16.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)

        busca_svm.fit(
            X_treino,
            y_treino,
        )

    tempo_total_busca_svm = (
        perf_counter() - inicio_busca_svm
    )

    resultados_grade_svm = busca_svm.cv_results_
    indice_melhor_svm = busca_svm.best_index_

    buscas_svm[identificador] = busca_svm
    melhores_pipelines_svm[identificador] = (
        busca_svm.best_estimator_
    )

    print("\nBusca concluída.")
    print("Parâmetros selecionados:", busca_svm.best_params_)
    print(
        "F1 macro médio na validação:",
        f"{busca_svm.best_score_:.4f}",
    )
    print(
        "Tempo total da busca, incluindo ajuste final:",
        f"{tempo_total_busca_svm:.2f} s",
    )

    # Preserva o cv_results_ completo.
    tabela_busca_completa_svm = pd.DataFrame(
        resultados_grade_svm
    )

    # Uma linha por configuração, com médias e DP.
    linhas_grade_svm = []

    for indice, parametros in enumerate(
        resultados_grade_svm["params"]
    ):
        linha = {
            "Modelo": nome_modelo,
            "Kernel": identificador,
            "C": parametros["modelo__C"],
            "Gamma": (
                parametros["modelo__gamma"]
                if identificador == "rbf"
                else "Não se aplica"
            ),
            "Peso das classes": (
                "Sem ponderação"
                if parametros["modelo__class_weight"] is None
                else "balanced"
            ),
            "Posição por F1 macro": int(
                resultados_grade_svm[
                    "rank_test_f1_macro"
                ][indice]
            ),
            "Selecionada": indice == indice_melhor_svm,
            "Tempo médio de ajuste (s)": (
                resultados_grade_svm["mean_fit_time"][indice]
            ),
            "Tempo médio de avaliação (s)": (
                resultados_grade_svm["mean_score_time"][indice]
            ),
        }

        for nome_metrica in metricas_cv:
            valores_treino = np.array([
                resultados_grade_svm[
                    f"split{divisao}_train_{nome_metrica}"
                ][indice]
                for divisao in range(len(divisoes_cv))
            ])

            valores_validacao = np.array([
                resultados_grade_svm[
                    f"split{divisao}_test_{nome_metrica}"
                ][indice]
                for divisao in range(len(divisoes_cv))
            ])

            linha[f"Treino - {nome_metrica}"] = (
                np.mean(valores_treino)
            )
            linha[f"Validação - {nome_metrica}"] = (
                np.mean(valores_validacao)
            )
            linha[f"DP validação - {nome_metrica}"] = (
                np.std(valores_validacao, ddof=1)
            )

        linha["Diferença F1 treino - validação (pp)"] = (
            linha["Treino - f1_macro"]
            - linha["Validação - f1_macro"]
        ) * 100

        linhas_grade_svm.append(linha)

    tabela_grade_svm = (
        pd.DataFrame(linhas_grade_svm)
        .sort_values(
            by="Posição por F1 macro",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # Resumo da configuração selecionada.
    linhas_resumo_svm_ajustado = []

    for nome_metrica in metricas_cv:
        valores_treino = np.array([
            resultados_grade_svm[
                f"split{divisao}_train_{nome_metrica}"
            ][indice_melhor_svm]
            for divisao in range(len(divisoes_cv))
        ])

        valores_validacao = np.array([
            resultados_grade_svm[
                f"split{divisao}_test_{nome_metrica}"
            ][indice_melhor_svm]
            for divisao in range(len(divisoes_cv))
        ])

        linhas_resumo_svm_ajustado.append({
            "Modelo": nome_modelo,
            "Métrica": nome_metrica,
            "Média no treino": np.mean(valores_treino),
            "Desvio padrão no treino": np.std(
                valores_treino,
                ddof=1,
            ),
            "Média na validação": np.mean(valores_validacao),
            "Desvio padrão na validação": np.std(
                valores_validacao,
                ddof=1,
            ),
            "Diferença treino - validação": (
                np.mean(valores_treino)
                - np.mean(valores_validacao)
            ),
        })

    tabela_resumo_svm_ajustado = pd.DataFrame(
        linhas_resumo_svm_ajustado
    )

    # Compara cada kernel com sua configuração inicial.
    comparacao_svm_ajuste = (
        tabela_inicial[
            ["Métrica", "Média na validação"]
        ]
        .rename(columns={
            "Média na validação": "Inicial - validação",
        })
        .merge(
            tabela_resumo_svm_ajustado[
                ["Métrica", "Média na validação"]
            ].rename(columns={
                "Média na validação": "Selecionada - validação",
            }),
            on="Métrica",
            validate="one_to_one",
        )
    )

    comparacao_svm_ajuste[
        "Diferença selecionada - inicial (pp)"
    ] = (
        comparacao_svm_ajuste["Selecionada - validação"]
        - comparacao_svm_ajuste["Inicial - validação"]
    ) * 100

    tabela_tempos_svm_ajustado = pd.DataFrame([{
        "Modelo": nome_modelo,
        "Kernel": identificador,
        "C selecionado": busca_svm.best_params_["modelo__C"],
        "Gamma selecionado": (
            busca_svm.best_params_["modelo__gamma"]
            if identificador == "rbf"
            else "Não se aplica"
        ),
        "Peso das classes": (
            "Sem ponderação"
            if busca_svm.best_params_[
                "modelo__class_weight"
            ] is None
            else "balanced"
        ),
        "Métrica de seleção": METRICA_PRINCIPAL,
        "Configurações": numero_configuracoes,
        "Divisões": len(divisoes_cv),
        "Tempo médio de ajuste (s)": resultados_grade_svm[
            "mean_fit_time"
        ][indice_melhor_svm],
        "Tempo médio de avaliação (s)": resultados_grade_svm[
            "mean_score_time"
        ][indice_melhor_svm],
        "Tempo do ajuste final no treino (s)": (
            busca_svm.refit_time_
        ),
        "Tempo total da busca com ajuste final (s)": (
            tempo_total_busca_svm
        ),
    }])

    tabelas_svm_busca_completa[identificador] = (
        tabela_busca_completa_svm
    )
    tabelas_svm_grade_ajuste[identificador] = tabela_grade_svm
    tabelas_svm_ajustadas_resumo[identificador] = (
        tabela_resumo_svm_ajustado
    )
    tabelas_svm_ajustadas_tempos[identificador] = (
        tabela_tempos_svm_ajustado
    )
    comparacoes_svm_ajuste[identificador] = (
        comparacao_svm_ajuste
    )

    print("\nConfigurações ordenadas pelo F1 macro:")
    print(tabela_grade_svm.round(4).to_string(index=False))

    print("\nResumo da configuração selecionada:")
    print(
        tabela_resumo_svm_ajustado.round(4).to_string(
            index=False
        )
    )

    print("\nComparação com a configuração inicial:")
    print(
        comparacao_svm_ajuste.round(4).to_string(index=False)
    )

    print("\nParâmetros e tempos:")
    print(
        tabela_tempos_svm_ajustado.round(6).to_string(
            index=False
        )
    )

    prefixo_svm_ajuste = f"svm_{identificador}"

    for tabela, sufixo in [
        (
            tabela_busca_completa_svm,
            "ajuste_busca_completa",
        ),
        (
            tabela_grade_svm,
            "ajuste_configuracoes",
        ),
        (
            tabela_resumo_svm_ajustado,
            "ajustado_validacao_cruzada_resumo",
        ),
        (
            tabela_tempos_svm_ajustado,
            "ajustado_tempos_parametros",
        ),
        (
            comparacao_svm_ajuste,
            "comparacao_inicial_ajustado",
        ),
    ]:
        caminho_svm_ajuste = (
            PASTA_RESULTADOS
            / f"{prefixo_svm_ajuste}_{sufixo}.csv"
        )

        tabela.to_csv(
            caminho_svm_ajuste,
            index=False,
            sep=";",
            encoding="utf-8-sig",
            decimal=",",
            float_format="%.6f",
        )

        print("\nTabela salva em:", caminho_svm_ajuste)

# Nomes explícitos para uso nos próximos blocos.
busca_svm_linear = buscas_svm["linear"]
busca_svm_rbf = buscas_svm["rbf"]

melhor_pipeline_svm_linear = melhores_pipelines_svm["linear"]
melhor_pipeline_svm_rbf = melhores_pipelines_svm["rbf"]

tabela_svm_linear_ajustado_resumo = (
    tabelas_svm_ajustadas_resumo["linear"]
)
tabela_svm_rbf_ajustado_resumo = (
    tabelas_svm_ajustadas_resumo["rbf"]
)

tabela_svm_linear_ajustado_tempos = (
    tabelas_svm_ajustadas_tempos["linear"]
)
tabela_svm_rbf_ajustado_tempos = (
    tabelas_svm_ajustadas_tempos["rbf"]
)

comparacao_svm_linear_ajuste = comparacoes_svm_ajuste["linear"]
comparacao_svm_rbf_ajuste = comparacoes_svm_ajuste["rbf"]

print("\nOs desvios padrão dos resumos usam ddof=1.")
print(
    "Os arquivos busca_completa preservam o cv_results_ "
    "original; os campos std_* seguem a convenção "
    "do scikit-learn."
)
print(
    "As posições das configurações são calculadas "
    "separadamente para cada kernel."
)
print(
    "As melhores médias foram selecionadas nas mesmas "
    "divisões usadas na busca. Podem ser otimistas "
    "e não constituem avaliação independente."
)
print(
    "As diferenças em relação às configurações iniciais "
    "são descritivas e não comprovam superioridade estatística."
)
print(
    "Os dois melhores Pipelines estão ajustados em todo "
    "o treino e disponíveis na memória desta execução."
)
print(
    "Os modelos ainda não foram salvos em arquivo "
    "nem escolhidos como vencedores entre todas as famílias."
)

print("\nO conjunto de teste reservado não foi utilizado.")
print("\nBloco 23 concluído.")

# %% ==========================================================
# BLOCO 24 — Ajuste do Gaussian Naive Bayes
# =============================================================

from time import perf_counter

from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.naive_bayes import GaussianNB

print("\n" + "=" * 70)
print("BLOCO 24 — AJUSTE DO GAUSSIAN NAIVE BAYES")
print("=" * 70)

# Mantém o mesmo Pipeline dos experimentos anteriores.
# priors=None estima as probabilidades das classes no treino.
#
# var_smoothing adiciona às variâncias uma fração da maior
# variância observada nas entradas transformadas.
# A grade explora diferentes intensidades dessa estabilização.
#
# Esse ajuste não altera a hipótese de independência
# condicional nem transforma indicadores em dados gaussianos.

pipeline_nb_busca = criar_pipeline(
    GaussianNB(
        priors=None,
    )
)

grade_nb = {
    "modelo__var_smoothing": [
        1e-12,
        1e-11,
        1e-10,
        1e-9,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
    ],
}

numero_configuracoes_nb = len(
    list(ParameterGrid(grade_nb))
)

print("\nModelo: Gaussian Naive Bayes")
print(
    "Valores de var_smoothing:",
    grade_nb["modelo__var_smoothing"],
)
print("Probabilidades das classes: estimadas no treino")
print("Métrica de seleção:", METRICA_PRINCIPAL)
print("Configurações:", numero_configuracoes_nb)
print("Divisões por configuração:", len(divisoes_cv))
print(
    "Ajustes na validação cruzada:",
    numero_configuracoes_nb * len(divisoes_cv),
)
print("Registros disponíveis no treino:", len(X_treino))

print(
    "\nO valor inicial var_smoothing=1e-9 "
    "está incluído na busca."
)
print(
    "O pré-processamento será aprendido dentro "
    "de cada divisão da validação cruzada."
)
print(
    "Após a seleção, o melhor Pipeline será ajustado "
    "em todo o conjunto de treino."
)
print("\nIniciando busca. Aguarde a conclusão...")

busca_nb = GridSearchCV(
    estimator=pipeline_nb_busca,
    param_grid=grade_nb,
    scoring=metricas_cv,
    refit=METRICA_PRINCIPAL,
    cv=divisoes_cv,
    return_train_score=True,
    n_jobs=1,
    verbose=2,
    error_score="raise",
)

inicio_busca_nb = perf_counter()

busca_nb.fit(
    X_treino,
    y_treino,
)

tempo_total_busca_nb = (
    perf_counter() - inicio_busca_nb
)

# Pipeline selecionado e ajustado em todo o treino.
melhor_pipeline_nb = busca_nb.best_estimator_

indice_melhor_nb = busca_nb.best_index_
resultados_grade_nb = busca_nb.cv_results_

print("\nBusca concluída.")
print("Parâmetros selecionados:", busca_nb.best_params_)
print(
    "F1 macro médio na validação:",
    f"{busca_nb.best_score_:.4f}",
)
print(
    "Tempo total da busca, incluindo ajuste final:",
    f"{tempo_total_busca_nb:.2f} s",
)
print(
    "Tempo do ajuste final em todo o treino:",
    f"{busca_nb.refit_time_:.2f} s",
)

# Preserva todos os resultados originais da busca.
tabela_nb_busca_completa = pd.DataFrame(
    resultados_grade_nb
)

# Uma linha por configuração.
linhas_grade_nb = []

for indice, parametros in enumerate(
    resultados_grade_nb["params"]
):
    linha = {
        "Var smoothing": parametros["modelo__var_smoothing"],
        "Posição por F1 macro": int(
            resultados_grade_nb[
                "rank_test_f1_macro"
            ][indice]
        ),
        "Selecionada": indice == indice_melhor_nb,
        "Tempo médio de ajuste (s)": resultados_grade_nb[
            "mean_fit_time"
        ][indice],
        "Tempo médio de avaliação (s)": resultados_grade_nb[
            "mean_score_time"
        ][indice],
    }

    for nome_metrica in metricas_cv:
        valores_treino = np.array([
            resultados_grade_nb[
                f"split{divisao}_train_{nome_metrica}"
            ][indice]
            for divisao in range(len(divisoes_cv))
        ])

        valores_validacao = np.array([
            resultados_grade_nb[
                f"split{divisao}_test_{nome_metrica}"
            ][indice]
            for divisao in range(len(divisoes_cv))
        ])

        linha[f"Treino - {nome_metrica}"] = (
            np.mean(valores_treino)
        )
        linha[f"Validação - {nome_metrica}"] = (
            np.mean(valores_validacao)
        )
        linha[f"DP validação - {nome_metrica}"] = (
            np.std(valores_validacao, ddof=1)
        )

    linha["Diferença F1 treino - validação (pp)"] = (
        linha["Treino - f1_macro"]
        - linha["Validação - f1_macro"]
    ) * 100

    linhas_grade_nb.append(linha)

tabela_nb_grade = (
    pd.DataFrame(linhas_grade_nb)
    .sort_values(
        by="Posição por F1 macro",
        kind="stable",
    )
    .reset_index(drop=True)
)

# Resumo da configuração selecionada.
linhas_nb_ajustado_resumo = []

for nome_metrica in metricas_cv:
    valores_treino = np.array([
        resultados_grade_nb[
            f"split{divisao}_train_{nome_metrica}"
        ][indice_melhor_nb]
        for divisao in range(len(divisoes_cv))
    ])

    valores_validacao = np.array([
        resultados_grade_nb[
            f"split{divisao}_test_{nome_metrica}"
        ][indice_melhor_nb]
        for divisao in range(len(divisoes_cv))
    ])

    linhas_nb_ajustado_resumo.append({
        "Modelo": "Gaussian Naive Bayes — ajustado",
        "Métrica": nome_metrica,
        "Média no treino": np.mean(valores_treino),
        "Desvio padrão no treino": np.std(
            valores_treino,
            ddof=1,
        ),
        "Média na validação": np.mean(valores_validacao),
        "Desvio padrão na validação": np.std(
            valores_validacao,
            ddof=1,
        ),
        "Diferença treino - validação": (
            np.mean(valores_treino)
            - np.mean(valores_validacao)
        ),
    })

tabela_nb_ajustado_resumo = pd.DataFrame(
    linhas_nb_ajustado_resumo
)

# Lê o resumo inicial já produzido pelo Bloco 17.
# Não exige um nome específico para sua variável em memória.
arquivo_nb_inicial_resumo = (
    PASTA_RESULTADOS
    / "nb_inicial_validacao_cruzada_resumo.csv"
)

if not arquivo_nb_inicial_resumo.is_file():
    raise FileNotFoundError(
        "Resumo inicial do Naive Bayes não encontrado: "
        f"{arquivo_nb_inicial_resumo}"
    )

tabela_nb_inicial_para_comparacao = pd.read_csv(
    arquivo_nb_inicial_resumo,
    sep=";",
    encoding="utf-8-sig",
    decimal=",",
)

comparacao_nb_ajuste = (
    tabela_nb_inicial_para_comparacao[
        ["Métrica", "Média na validação"]
    ]
    .rename(columns={
        "Média na validação": "Inicial - validação",
    })
    .merge(
        tabela_nb_ajustado_resumo[
            ["Métrica", "Média na validação"]
        ].rename(columns={
            "Média na validação": "Selecionada - validação",
        }),
        on="Métrica",
        validate="one_to_one",
    )
)

comparacao_nb_ajuste[
    "Diferença selecionada - inicial (pp)"
] = (
    comparacao_nb_ajuste["Selecionada - validação"]
    - comparacao_nb_ajuste["Inicial - validação"]
) * 100

tabela_nb_ajustado_tempos = pd.DataFrame([{
    "Modelo": "Gaussian Naive Bayes — ajustado",
    "Var smoothing selecionado": busca_nb.best_params_[
        "modelo__var_smoothing"
    ],
    "Probabilidades das classes": "Estimadas no treino",
    "Métrica de seleção": METRICA_PRINCIPAL,
    "Configurações": numero_configuracoes_nb,
    "Divisões": len(divisoes_cv),
    "Tempo médio de ajuste (s)": resultados_grade_nb[
        "mean_fit_time"
    ][indice_melhor_nb],
    "Tempo médio de avaliação (s)": resultados_grade_nb[
        "mean_score_time"
    ][indice_melhor_nb],
    "Tempo do ajuste final no treino (s)": (
        busca_nb.refit_time_
    ),
    "Tempo total da busca com ajuste final (s)": (
        tempo_total_busca_nb
    ),
}])

# Usa notação científica para não mostrar valores pequenos
# de var_smoothing como zero na apresentação.
tabela_nb_grade_exibicao = tabela_nb_grade.copy()
tabela_nb_grade_exibicao["Var smoothing"] = (
    tabela_nb_grade_exibicao["Var smoothing"]
    .map(lambda valor: f"{valor:.1e}")
)

tabela_nb_tempos_exibicao = tabela_nb_ajustado_tempos.copy()
tabela_nb_tempos_exibicao["Var smoothing selecionado"] = (
    tabela_nb_tempos_exibicao["Var smoothing selecionado"]
    .map(lambda valor: f"{valor:.1e}")
)

print("\nConfigurações ordenadas pelo F1 macro:")
print(
    tabela_nb_grade_exibicao.round(4).to_string(index=False)
)

print("\nResumo da configuração selecionada:")
print(
    tabela_nb_ajustado_resumo.round(4).to_string(index=False)
)

print("\nComparação com a configuração inicial:")
print(
    comparacao_nb_ajuste.round(4).to_string(index=False)
)

print("\nParâmetros e tempos:")
print(
    tabela_nb_tempos_exibicao.round(6).to_string(index=False)
)

# Usa 12 algarismos significativos para preservar os valores
# pequenos de var_smoothing nos CSVs.
# Mantém separador, codificação e decimal do projeto.
for tabela, nome_arquivo in [
    (
        tabela_nb_busca_completa,
        "nb_ajuste_busca_completa.csv",
    ),
    (
        tabela_nb_grade,
        "nb_ajuste_configuracoes.csv",
    ),
    (
        tabela_nb_ajustado_resumo,
        "nb_ajustado_validacao_cruzada_resumo.csv",
    ),
    (
        tabela_nb_ajustado_tempos,
        "nb_ajustado_tempos_parametros.csv",
    ),
    (
        comparacao_nb_ajuste,
        "comparacao_nb_inicial_ajustado.csv",
    ),
]:
    caminho_nb_ajuste = PASTA_RESULTADOS / nome_arquivo

    tabela.to_csv(
        caminho_nb_ajuste,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        float_format="%.12g",
    )

    print("\nTabela salva em:", caminho_nb_ajuste)

print("\nOs desvios padrão dos resumos usam ddof=1.")
print(
    "O arquivo busca_completa preserva o cv_results_ "
    "original; os campos std_* seguem a convenção "
    "do scikit-learn."
)
print(
    "A busca varia somente var_smoothing e mantém "
    "o pré-processamento comum dos experimentos."
)
print(
    "A estabilização das variâncias não elimina "
    "as limitações das hipóteses do Gaussian Naive Bayes."
)
print(
    "A melhor média foi selecionada nas mesmas divisões "
    "usadas na busca. Pode ser otimista e não constitui "
    "uma avaliação independente."
)
print(
    "A diferença em relação à configuração inicial "
    "é descritiva; não comprova superioridade estatística."
)
print(
    "melhor_pipeline_nb está ajustado em todo o treino "
    "e disponível na memória desta execução."
)
print(
    "O modelo ainda não foi salvo em arquivo nem escolhido "
    "como vencedor entre todas as famílias."
)

print("\nO conjunto de teste reservado não foi utilizado.")
print("\nBloco 24 concluído.")

# ============================================================
# BLOCO 25 — CONSOLIDAÇÃO DOS RESULTADOS
# ============================================================
# Utiliza somente os CSVs de validação cruzada já produzidos.
# Não treina modelos nem acessa o conjunto de teste reservado.

print("\n" + "=" * 70)
print("BLOCO 25 — CONSOLIDAÇÃO DOS RESULTADOS")
print("=" * 70)

modelos_bloco25 = [
    ("Regressão Logística", "logistica_inicial", "logistica_ajustada"),
    ("KNN", "knn_inicial", "knn_ajustado"),
    ("Árvore", "arvore_inicial", "arvore_ajustada"),
    ("Random Forest", "forest_inicial", "forest_ajustada"),
    ("SVM linear", "svm_linear_inicial", "svm_linear_ajustado"),
    ("SVM RBF", "svm_rbf_inicial", "svm_rbf_ajustado"),
    ("Gaussian Naive Bayes", "nb_inicial", "nb_ajustado"),
]

metricas_bloco25 = [
    "acuracia",
    "precisao_macro",
    "recall_macro",
    "f1_macro",
]


def ler_resumo_bloco25(prefixo):
    caminho = (
        PASTA_RESULTADOS
        / f"{prefixo}_validacao_cruzada_resumo.csv"
    )

    tabela = pd.read_csv(
        caminho,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
    )

    colunas = [
        "Métrica",
        "Média no treino",
        "Desvio padrão no treino",
        "Média na validação",
        "Desvio padrão na validação",
    ]

    faltantes = set(colunas) - set(tabela.columns)
    if faltantes:
        raise ValueError(
            f"Colunas ausentes em {caminho.name}: {faltantes}"
        )

    if tabela["Métrica"].duplicated().any():
        raise ValueError(
            f"Métricas duplicadas em {caminho.name}."
        )

    tabela = tabela.set_index("Métrica")

    if set(tabela.index) != set(metricas_bloco25):
        raise ValueError(
            f"Métricas inesperadas em {caminho.name}."
        )

    numericas = colunas[1:]
    tabela[numericas] = tabela[numericas].apply(
        pd.to_numeric,
        errors="raise",
    )

    if tabela[numericas].isna().any().any():
        raise ValueError(
            f"Valores ausentes em {caminho.name}."
        )

    if not tabela[numericas].apply(
        lambda coluna: coluna.between(0, 1).all()
    ).all():
        raise ValueError(
            f"Valores fora de 0 a 1 em {caminho.name}."
        )

    return tabela


linhas_bloco25 = []

for nome, prefixo_inicial, prefixo_ajustado in modelos_bloco25:
    inicial = ler_resumo_bloco25(prefixo_inicial)
    ajustado = ler_resumo_bloco25(prefixo_ajustado)

    for metrica in metricas_bloco25:
        valor_inicial = inicial.loc[metrica]
        valor_ajustado = ajustado.loc[metrica]

        linhas_bloco25.append({
            "Modelo": nome,
            "Métrica": metrica,
            "Média treino inicial":
                valor_inicial["Média no treino"],
            "DP treino inicial":
                valor_inicial["Desvio padrão no treino"],
            "Média validação inicial":
                valor_inicial["Média na validação"],
            "DP validação inicial":
                valor_inicial["Desvio padrão na validação"],
            "Média treino ajustado":
                valor_ajustado["Média no treino"],
            "DP treino ajustado":
                valor_ajustado["Desvio padrão no treino"],
            "Média validação ajustado":
                valor_ajustado["Média na validação"],
            "DP validação ajustado":
                valor_ajustado["Desvio padrão na validação"],
            "Ganho validação (pp)": 100 * (
                valor_ajustado["Média na validação"]
                - valor_inicial["Média na validação"]
            ),
            "Diferença treino-validação inicial (pp)": 100 * (
                valor_inicial["Média no treino"]
                - valor_inicial["Média na validação"]
            ),
            "Diferença treino-validação ajustado (pp)": 100 * (
                valor_ajustado["Média no treino"]
                - valor_ajustado["Média na validação"]
            ),
        })

tabela_consolidada_bloco25 = pd.DataFrame(linhas_bloco25)

tabela_f1_bloco25 = (
    tabela_consolidada_bloco25.loc[
        tabela_consolidada_bloco25["Métrica"] == "f1_macro"
    ]
    .sort_values(
        "Média validação ajustado",
        ascending=False,
        kind="stable",
    )
    .reset_index(drop=True)
)

arquivo_consolidado_bloco25 = (
    PASTA_RESULTADOS
    / "comparacao_iniciais_ajustados_consolidada.csv"
)
arquivo_f1_bloco25 = (
    PASTA_RESULTADOS
    / "comparacao_iniciais_ajustados_f1_macro.csv"
)
arquivo_figura_bloco25 = (
    PASTA_RESULTADOS
    / "figura_05_comparacao_iniciais_ajustados.png"
)

for tabela, destino in [
    (tabela_consolidada_bloco25, arquivo_consolidado_bloco25),
    (tabela_f1_bloco25, arquivo_f1_bloco25),
]:
    tabela.to_csv(
        destino,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        index=False,
        float_format="%.6f",
    )

print("\nComparação pelo F1 macro:")
print(
    tabela_f1_bloco25[[
        "Modelo",
        "Média validação inicial",
        "Média validação ajustado",
        "DP validação ajustado",
        "Ganho validação (pp)",
        "Diferença treino-validação ajustado (pp)",
    ]].to_string(
        index=False,
        float_format=lambda valor: f"{valor:.4f}",
    )
)

dummy_bloco25 = ler_resumo_bloco25("dummy")
f1_dummy_bloco25 = dummy_bloco25.loc[
    "f1_macro", "Média na validação"
]

figura_bloco25, eixo_bloco25 = plt.subplots(
    figsize=(12, 7)
)

posicoes_bloco25 = list(range(len(tabela_f1_bloco25)))
deslocamento_bloco25 = 0.16

eixo_bloco25.errorbar(
    tabela_f1_bloco25["Média validação inicial"],
    [posicao - deslocamento_bloco25
     for posicao in posicoes_bloco25],
    xerr=tabela_f1_bloco25["DP validação inicial"],
    fmt="o",
    color="#64748b",
    capsize=4,
    label="Inicial: validação ± 1 DP",
)

eixo_bloco25.errorbar(
    tabela_f1_bloco25["Média validação ajustado"],
    [posicao + deslocamento_bloco25
     for posicao in posicoes_bloco25],
    xerr=tabela_f1_bloco25["DP validação ajustado"],
    fmt="o",
    color="#2563eb",
    capsize=4,
    label="Ajustado: validação ± 1 DP",
)

eixo_bloco25.scatter(
    tabela_f1_bloco25["Média treino ajustado"],
    [posicao + deslocamento_bloco25
     for posicao in posicoes_bloco25],
    marker="D",
    color="#d97706",
    s=35,
    label="Ajustado: média no treino",
    zorder=3,
)

eixo_bloco25.axvline(
    f1_dummy_bloco25,
    color="#9333ea",
    linestyle="--",
    linewidth=1.3,
    label=f"Dummy: F1 validação = {f1_dummy_bloco25:.4f}",
)

eixo_bloco25.set_yticks(posicoes_bloco25)
eixo_bloco25.set_yticklabels(tabela_f1_bloco25["Modelo"])
eixo_bloco25.invert_yaxis()
eixo_bloco25.set_xlim(0, 1.03)
eixo_bloco25.set_xlabel("F1 macro")
eixo_bloco25.set_title(
    "Figura 5 — Configurações iniciais e ajustadas\n"
    "Ordenação pela média de validação dos ajustados"
)
eixo_bloco25.grid(axis="x", alpha=0.25)
eixo_bloco25.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.13),
    ncol=2,
    frameon=False,
)

figura_bloco25.text(
    0.5,
    0.02,
    "Barras: ± 1 desvio padrão entre divisões; "
    "não são intervalos de confiança.\n"
    "A seleção usou as mesmas divisões: "
    "os resultados ajustados podem ser otimistas.",
    ha="center",
    fontsize=9,
)

figura_bloco25.tight_layout(rect=(0, 0.16, 1, 1))
figura_bloco25.savefig(
    arquivo_figura_bloco25,
    dpi=300,
    bbox_inches="tight",
)

print("\nArquivos gerados:")
print(arquivo_consolidado_bloco25)
print(arquivo_f1_bloco25)
print(arquivo_figura_bloco25)

print(
    "\nAs diferenças são descritivas e não comprovam "
    "superioridade estatística."
)
print(
    "Os valores foram lidos dos CSVs e podem apresentar "
    "pequenas diferenças decorrentes do arredondamento."
)
print(
    "Esta consolidação não escolhe automaticamente "
    "o modelo final."
)
print(
    "Nenhuma busca foi repetida neste bloco. "
    "O conjunto de teste reservado não foi utilizado."
)

plt.show()
plt.close(figura_bloco25)

print("\nBloco 25 concluído.")

# ============================================================
# BLOCO 26 — ANÁLISE DOS ERROS POR CLASSE NA VALIDAÇÃO
# ============================================================
# Análise descritiva com os parâmetros selecionados anteriormente.
# Não utiliza o conjunto de teste reservado.
# Não realiza novas buscas de hiperparâmetros.

from sklearn.base import clone
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from time import perf_counter
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

print("\n" + "=" * 70)
print("BLOCO 26 — ANÁLISE DOS ERROS POR CLASSE")
print("=" * 70)

modelos_bloco26 = [
    ("logistica", "Regressão Logística", melhor_pipeline_logistica),
    ("knn", "KNN", melhor_pipeline_knn),
    ("arvore", "Árvore", melhor_pipeline_arvore),
    ("forest", "Random Forest", melhor_pipeline_forest),
    ("svm_linear", "SVM linear", melhor_pipeline_svm_linear),
    ("svm_rbf", "SVM RBF", melhor_pipeline_svm_rbf),
    ("nb", "Gaussian Naive Bayes", melhor_pipeline_nb),
]

if not X_treino.index.equals(y_treino.index):
    raise ValueError("Os índices de X_treino e y_treino diferem.")

if set(y_treino.unique()) != set(ordem_classes):
    raise ValueError("As classes do treino diferem da ordem esperada.")

# Confere se cada registro participa uma vez da validação.
cobertura_bloco26 = np.zeros(len(y_treino), dtype=int)

for treino_pos, validacao_pos in divisoes_cv:
    if np.intersect1d(treino_pos, validacao_pos).size:
        raise ValueError("Há sobreposição dentro de uma divisão.")

    np.add.at(cobertura_bloco26, validacao_pos, 1)

if not np.all(cobertura_bloco26 == 1):
    raise ValueError(
        "Cada registro deve aparecer exatamente uma vez na validação."
    )

metricas_classe_bloco26 = []
metricas_agregadas_bloco26 = []
tempos_bloco26 = []
matrizes_bloco26 = {}
previsoes_bloco26 = {}

inicio_bloco26 = perf_counter()

for identificador, nome, pipeline in modelos_bloco26:
    print(f"\nModelo: {nome}")
    inicio_modelo = perf_counter()

    previsoes = np.empty(len(y_treino), dtype=object)
    numero_divisao = np.zeros(len(y_treino), dtype=int)

    for divisao, (treino_pos, validacao_pos) in enumerate(
        divisoes_cv, start=1
    ):
        print(
            f"  Divisão {divisao}/{len(divisoes_cv)}...",
            flush=True,
        )

        # clone remove o estado treinado e preserva os parâmetros.
        # O pré-processamento também é reaprendido nesta divisão.
        pipeline_divisao = clone(pipeline)

        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)

            pipeline_divisao.fit(
                X_treino.iloc[treino_pos],
                y_treino.iloc[treino_pos],
            )

            previsoes[validacao_pos] = pipeline_divisao.predict(
                X_treino.iloc[validacao_pos]
            )

        numero_divisao[validacao_pos] = divisao

    if not set(previsoes).issubset(set(ordem_classes)):
        raise ValueError(f"Previsões inesperadas para {nome}.")

    relatorio = classification_report(
        y_treino,
        previsoes,
        labels=ordem_classes,
        output_dict=True,
        zero_division=0,
    )

    for classe in ordem_classes:
        resultado = relatorio[classe]

        metricas_classe_bloco26.append({
            "Modelo": nome,
            "Classe": classe,
            "Precisão": resultado["precision"],
            "Recall": resultado["recall"],
            "F1": resultado["f1-score"],
            "Suporte": int(resultado["support"]),
        })

    metricas_agregadas_bloco26.append({
        "Modelo": nome,
        "Acurácia OOF agregada": relatorio["accuracy"],
        "Precisão macro OOF agregada":
            relatorio["macro avg"]["precision"],
        "Recall macro OOF agregado":
            relatorio["macro avg"]["recall"],
        "F1 macro OOF agregado":
            relatorio["macro avg"]["f1-score"],
        "Registros": len(y_treino),
    })

    matriz = confusion_matrix(
        y_treino,
        previsoes,
        labels=ordem_classes,
    )

    if matriz.sum() != len(y_treino):
        raise ValueError(f"Contagem inconsistente para {nome}.")

    matriz_percentual = (
        matriz / matriz.sum(axis=1, keepdims=True) * 100
    )

    matrizes_bloco26[identificador] = matriz
    previsoes_bloco26[identificador] = previsoes.copy()

    tabela_previsoes = pd.DataFrame({
        "Indice_original": y_treino.index,
        "Divisao_validacao": numero_divisao,
        "Classe_real": y_treino.to_numpy(),
        "Classe_prevista": previsoes,
        "Acertou": y_treino.to_numpy() == previsoes,
    })

    tabela_matriz = pd.DataFrame(
        matriz,
        index=pd.Index(ordem_classes, name="Classe real"),
        columns=pd.Index(ordem_classes, name="Classe prevista"),
    )

    tabela_percentual = pd.DataFrame(
        matriz_percentual,
        index=pd.Index(ordem_classes, name="Classe real"),
        columns=pd.Index(ordem_classes, name="Classe prevista"),
    )

    for tabela, sufixo, salvar_indice in [
        (tabela_previsoes, "previsoes_oof", False),
        (tabela_matriz, "matriz_confusao_oof", True),
        (tabela_percentual, "matriz_confusao_oof_percentual", True),
    ]:
        tabela.to_csv(
            PASTA_RESULTADOS / f"{identificador}_{sufixo}.csv",
            sep=";",
            encoding="utf-8-sig",
            decimal=",",
            index=salvar_indice,
            float_format="%.6f",
        )

    tempos_bloco26.append({
        "Modelo": nome,
        "Divisões": len(divisoes_cv),
        "Tempo OOF total (s)": perf_counter() - inicio_modelo,
    })

    print(f"  Previsões de validação concluídas: {len(previsoes)}")

tabela_classes_bloco26 = pd.DataFrame(metricas_classe_bloco26)
tabela_agregada_bloco26 = pd.DataFrame(metricas_agregadas_bloco26)
tabela_tempos_bloco26 = pd.DataFrame(tempos_bloco26)

for tabela, arquivo in [
    (
        tabela_classes_bloco26,
        "metricas_por_classe_validacao_oof.csv",
    ),
    (
        tabela_agregada_bloco26,
        "metricas_agregadas_validacao_oof.csv",
    ),
    (
        tabela_tempos_bloco26,
        "tempos_validacao_oof.csv",
    ),
]:
    tabela.to_csv(
        PASTA_RESULTADOS / arquivo,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        index=False,
        float_format="%.6f",
    )

print("\nMétricas por classe — previsões de validação reunidas:")
print(
    tabela_classes_bloco26.to_string(
        index=False,
        float_format=lambda valor: f"{valor:.4f}",
    )
)

print("\nMétricas agregadas das previsões de validação:")
print(
    tabela_agregada_bloco26.to_string(
        index=False,
        float_format=lambda valor: f"{valor:.4f}",
    )
)

# Figura 6 — Matrizes com espaçamento e rótulos legíveis.
figura_bloco26, eixos_bloco26 = plt.subplots(
    3, 3,
    figsize=(16, 16),
)

figura_bloco26.canvas.manager.set_window_title(
    "Figura 6 — Matrizes de confusão"
)

for eixo, (identificador, nome, _) in zip(
    eixos_bloco26.flat, modelos_bloco26
):
    matriz = matrizes_bloco26[identificador]
    matriz_proporcao = (
        matriz / matriz.sum(axis=1, keepdims=True)
    )

    exibicao = ConfusionMatrixDisplay(
        confusion_matrix=matriz_proporcao,
        display_labels=ordem_classes,
    )

    exibicao.plot(
        ax=eixo,
        cmap="Blues",
        values_format=".1%",
        colorbar=False,
        im_kw={"vmin": 0, "vmax": 1},
    )

    eixo.set_title(nome, fontsize=12, pad=14)
    eixo.set_xlabel("Classe prevista", fontsize=10, labelpad=8)
    eixo.set_ylabel("Classe real", fontsize=10, labelpad=8)

    plt.setp(
        eixo.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
        fontsize=10,
    )
    eixo.tick_params(axis="y", labelsize=10)

    for texto in exibicao.text_.flat:
        texto.set_fontsize(10)

for eixo in list(eixos_bloco26.flat)[len(modelos_bloco26):]:
    eixo.axis("off")

figura_bloco26.suptitle(
    "Figura 6 — Matrizes de confusão na validação\n"
    "Configurações ajustadas — percentuais por classe real",
    fontsize=16,
    y=0.97,
)

# Espaço explícito entre linhas para acomodar os rótulos.
figura_bloco26.subplots_adjust(
    left=0.09,
    right=0.97,
    top=0.89,
    bottom=0.12,
    wspace=0.55,
    hspace=0.85,
)

figura_bloco26.text(
    0.5,
    0.035,
    "Cada linha soma 100%. Diagonal: acertos; demais células: erros.\n"
    "Parâmetros selecionados nas mesmas divisões: "
    "análise descritiva sujeita ao viés de seleção.",
    ha="center",
    va="bottom",
    fontsize=10,
)

arquivo_figura_bloco26 = (
    PASTA_RESULTADOS
    / "figura_06_matrizes_confusao_validacao_oof.png"
)

figura_bloco26.savefig(
    arquivo_figura_bloco26,
    dpi=300,
    bbox_inches="tight",
)

print("\nResultados salvos em:", PASTA_RESULTADOS)
print("Figura salva:", arquivo_figura_bloco26)
print(
    "\nOOF significa previsão fora da divisão usada no ajuste."
)
print(
    "As métricas agregadas reúnem todas as previsões OOF; "
    "podem diferir da média das cinco métricas do Bloco 25."
)
print(
    "A seleção dos parâmetros utilizou essas mesmas divisões. "
    "Esta análise não é uma avaliação independente."
)
print(
    "Não houve nova busca de parâmetros neste bloco "
    "nem alteração dos pipelines selecionados."
)
print("O conjunto de teste reservado não foi utilizado.")
print(
    f"Tempo total do Bloco 26: "
    f"{perf_counter() - inicio_bloco26:.2f} segundos."
)

plt.show()
plt.close(figura_bloco26)

print("\nBloco 26 concluído.")

# ============================================================
# BLOCO 27 — ESCOLHA REGISTRADA E AVALIAÇÃO FINAL NO TESTE
# ============================================================
# Modelo principal escolhido pela validação: Regressão Logística.
# Os demais modelos entram como comparação acadêmica.
# Não alterar parâmetros ou escolha com base no resultado do teste.

import json
from datetime import datetime, timezone
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

print("\n" + "=" * 70)
print("BLOCO 27 — AVALIAÇÃO FINAL NO TESTE RESERVADO")
print("=" * 70)

modelos_bloco27 = [
    ("logistica", "Regressão Logística", melhor_pipeline_logistica),
    ("knn", "KNN", melhor_pipeline_knn),
    ("arvore", "Árvore", melhor_pipeline_arvore),
    ("forest", "Random Forest", melhor_pipeline_forest),
    ("svm_linear", "SVM linear", melhor_pipeline_svm_linear),
    ("svm_rbf", "SVM RBF", melhor_pipeline_svm_rbf),
    ("nb", "Gaussian Naive Bayes", melhor_pipeline_nb),
]

# Verificações estruturais antes de calcular métricas.
if not X_teste.index.equals(y_teste.index):
    raise ValueError("Os índices de X_teste e y_teste diferem.")

if not X_treino.columns.equals(X_teste.columns):
    raise ValueError("As colunas de treino e teste diferem.")

if not X_treino.index.intersection(X_teste.index).empty:
    raise ValueError("Há índices compartilhados entre treino e teste.")

for identificador, nome, pipeline in modelos_bloco27:
    check_is_fitted(pipeline.named_steps["preprocessamento"])
    check_is_fitted(pipeline.named_steps["modelo"])

# Registra a decisão ANTES de gerar previsões no teste.
# Um registro existente deve concordar com a decisão atual.
decisao_bloco27 = {
    "modelo_principal": "logistica",
    "nome_modelo_principal": "Regressão Logística",
    "criterio": "Maior média de F1 macro na validação cruzada",
    "origem_criterio": "Bloco 25",
    "modelos_comparados": [
        identificador
        for identificador, _, _ in modelos_bloco27
    ],
    "parametros_modelos": {
        identificador: pipeline.named_steps["modelo"].get_params(
            deep=False
        )
        for identificador, _, pipeline in modelos_bloco27
    },
    "regra": (
        "Não reajustar parâmetros nem trocar o modelo principal "
        "com base nos resultados do teste."
    ),
}

# Converte tipos numéricos de bibliotecas para tipos JSON.
decisao_bloco27 = json.loads(
    json.dumps(
        decisao_bloco27,
        ensure_ascii=False,
        default=str,
    )
)

arquivo_decisao_bloco27 = (
    PASTA_RESULTADOS / "decisao_modelos_antes_teste.json"
)

if arquivo_decisao_bloco27.exists():
    with arquivo_decisao_bloco27.open(
        "r", encoding="utf-8"
    ) as arquivo:
        registro_anterior_bloco27 = json.load(arquivo)

    for chave, valor in decisao_bloco27.items():
        if registro_anterior_bloco27.get(chave) != valor:
            raise ValueError(
                "A decisão ou os parâmetros diferem do registro "
                "anterior. Interrompido antes das previsões."
            )

    print("\nRegistro de decisão anterior preservado.")
else:
    registro_bloco27 = {
        **decisao_bloco27,
        "registrado_em_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with arquivo_decisao_bloco27.open(
        "x", encoding="utf-8"
    ) as arquivo:
        json.dump(
            registro_bloco27,
            arquivo,
            ensure_ascii=False,
            indent=2,
        )

    print("\nDecisão registrada antes das previsões.")

print("Modelo principal: Regressão Logística.")
print("Critério: maior média de F1 macro na validação.")
print("Registro:", arquivo_decisao_bloco27)

if set(y_teste.unique()) != set(ordem_classes):
    raise ValueError("As classes do teste diferem das esperadas.")

metricas_gerais_bloco27 = []
metricas_classes_bloco27 = []
matrizes_bloco27 = {}
previsoes_bloco27 = {}

for identificador, nome, pipeline in modelos_bloco27:
    print(f"\nAvaliando no teste: {nome}", flush=True)

    # best_estimator_ foi ajustado em todo o treino nas buscas.
    # Aqui apenas prevemos: não há fit nem fit_transform.
    previsoes = pipeline.predict(X_teste)

    if len(previsoes) != len(y_teste):
        raise ValueError(f"Quantidade incorreta de previsões: {nome}.")

    if not set(previsoes).issubset(set(ordem_classes)):
        raise ValueError(f"Classes previstas inesperadas: {nome}.")

    relatorio = classification_report(
        y_teste,
        previsoes,
        labels=ordem_classes,
        output_dict=True,
        zero_division=0,
    )

    metricas_gerais_bloco27.append({
        "Modelo": nome,
        "Papel": (
            "Principal"
            if identificador == "logistica"
            else "Comparação"
        ),
        "Acurácia teste": accuracy_score(y_teste, previsoes),
        "Precisão macro teste":
            relatorio["macro avg"]["precision"],
        "Recall macro teste":
            relatorio["macro avg"]["recall"],
        "F1 macro teste":
            relatorio["macro avg"]["f1-score"],
        "Registros": len(y_teste),
    })

    for classe in ordem_classes:
        resultado = relatorio[classe]

        metricas_classes_bloco27.append({
            "Modelo": nome,
            "Classe": classe,
            "Precisão teste": resultado["precision"],
            "Recall teste": resultado["recall"],
            "F1 teste": resultado["f1-score"],
            "Suporte": int(resultado["support"]),
        })

    matriz = confusion_matrix(
        y_teste,
        previsoes,
        labels=ordem_classes,
    )

    if matriz.sum() != len(y_teste):
        raise ValueError(f"Contagem inconsistente: {nome}.")

    matriz_percentual = (
        matriz / matriz.sum(axis=1, keepdims=True) * 100
    )

    matrizes_bloco27[identificador] = matriz
    previsoes_bloco27[identificador] = previsoes.copy()

    tabela_previsoes = pd.DataFrame({
        "Indice_original": y_teste.index,
        "Classe_real": y_teste.to_numpy(),
        "Classe_prevista": previsoes,
        "Acertou": y_teste.to_numpy() == previsoes,
    })

    tabela_matriz = pd.DataFrame(
        matriz,
        index=pd.Index(ordem_classes, name="Classe real"),
        columns=pd.Index(ordem_classes, name="Classe prevista"),
    )

    tabela_percentual = pd.DataFrame(
        matriz_percentual,
        index=pd.Index(ordem_classes, name="Classe real"),
        columns=pd.Index(ordem_classes, name="Classe prevista"),
    )

    for tabela, sufixo, salvar_indice in [
        (tabela_previsoes, "previsoes_teste", False),
        (tabela_matriz, "matriz_confusao_teste", True),
        (tabela_percentual, "matriz_confusao_teste_percentual", True),
    ]:
        tabela.to_csv(
            PASTA_RESULTADOS / f"{identificador}_{sufixo}.csv",
            sep=";",
            encoding="utf-8-sig",
            decimal=",",
            index=salvar_indice,
            float_format="%.6f",
        )

tabela_geral_bloco27 = pd.DataFrame(metricas_gerais_bloco27)
tabela_classes_bloco27 = pd.DataFrame(metricas_classes_bloco27)

# Mantém a ordem definida antes do teste, sem escolher pelo teste.
for tabela, nome_arquivo in [
    (tabela_geral_bloco27, "metricas_gerais_teste.csv"),
    (tabela_classes_bloco27, "metricas_por_classe_teste.csv"),
]:
    tabela.to_csv(
        PASTA_RESULTADOS / nome_arquivo,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        index=False,
        float_format="%.6f",
    )

print("\nMétricas gerais no teste reservado:")
print(
    tabela_geral_bloco27.to_string(
        index=False,
        float_format=lambda valor: f"{valor:.4f}",
    )
)

print("\nMétricas por classe no teste reservado:")
print(
    tabela_classes_bloco27.to_string(
        index=False,
        float_format=lambda valor: f"{valor:.4f}",
    )
)

# Figura 7 — Mesmo padrão de espaçamento da Figura 6 salva.
figura_bloco27, eixos_bloco27 = plt.subplots(
    3, 3,
    figsize=(16, 16),
)

figura_bloco27.canvas.manager.set_window_title(
    "Figura 7 — Matrizes de confusão no teste"
)

for eixo, (identificador, nome, _) in zip(
    eixos_bloco27.flat, modelos_bloco27
):
    matriz = matrizes_bloco27[identificador]
    matriz_proporcao = (
        matriz / matriz.sum(axis=1, keepdims=True)
    )

    exibicao = ConfusionMatrixDisplay(
        confusion_matrix=matriz_proporcao,
        display_labels=ordem_classes,
    )

    exibicao.plot(
        ax=eixo,
        cmap="Blues",
        values_format=".1%",
        colorbar=False,
        im_kw={"vmin": 0, "vmax": 1},
    )

    titulo = (
        f"{nome} (principal)"
        if identificador == "logistica"
        else nome
    )

    eixo.set_title(titulo, fontsize=12, pad=14)
    eixo.set_xlabel("Classe prevista", fontsize=10, labelpad=8)
    eixo.set_ylabel("Classe real", fontsize=10, labelpad=8)

    plt.setp(
        eixo.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
        fontsize=10,
    )
    eixo.tick_params(axis="y", labelsize=10)

    for texto in exibicao.text_.flat:
        texto.set_fontsize(10)

for eixo in list(eixos_bloco27.flat)[len(modelos_bloco27):]:
    eixo.axis("off")

figura_bloco27.suptitle(
    "Figura 7 — Matrizes de confusão no teste reservado\n"
    "Configurações ajustadas — percentuais por classe real",
    fontsize=16,
    y=0.97,
)

figura_bloco27.subplots_adjust(
    left=0.09,
    right=0.97,
    top=0.89,
    bottom=0.12,
    wspace=0.55,
    hspace=0.85,
)

figura_bloco27.text(
    0.5,
    0.035,
    "Cada linha soma 100%. Diagonal: acertos; demais células: erros.\n"
    "Modelo principal definido pela validação antes desta avaliação. "
    "O teste não orienta novos ajustes.",
    ha="center",
    va="bottom",
    fontsize=10,
)

arquivo_figura_bloco27 = (
    PASTA_RESULTADOS
    / "figura_07_matrizes_confusao_teste.png"
)

figura_bloco27.savefig(
    arquivo_figura_bloco27,
    dpi=300,
    bbox_inches="tight",
)

print("\nResultados salvos em:", PASTA_RESULTADOS)
print("Figura salva:", arquivo_figura_bloco27)
print(
    "\nO teste reservado foi utilizado neste bloco "
    "para a avaliação final."
)
print(
    "Nenhum modelo foi treinado novamente e nenhuma "
    "busca de parâmetros foi realizada neste bloco."
)
print(
    "A Regressão Logística permanece como modelo principal, "
    "conforme a decisão registrada antes das previsões."
)
print(
    "Não interpretar pequenas diferenças no teste "
    "como comprovação de superioridade estatística."
)
print(
    "Limitação já identificada: houve exploração do dataset "
    "completo antes da divisão treino/teste."
)
print(
    "Os pipelines ainda não foram salvos em arquivos "
    "para uso no dashboard."
)

plt.show()
plt.close(figura_bloco27)

print("\nBloco 27 concluído.")


# ============================================================
# BLOCO 28 — SALVAMENTO DOS PIPELINES E METADADOS
# ============================================================

import json
import platform
from importlib.metadata import version
from tempfile import TemporaryDirectory

import joblib
import numpy as np


print("\nBLOCO 28 — Salvamento dos pipelines e metadados")

pasta_modelos_bloco28 = PASTA_PROJETO / "modelos"

# Recupera a decisão registrada antes da avaliação no teste.
with arquivo_decisao_bloco27.open("r", encoding="utf-8") as arquivo:
    decisao_registrada_bloco28 = json.load(arquivo)

# A mesma amostra de treino é usada para comparar todos os pipelines.
amostra_treino_bloco28 = X_treino.sample(
    n=min(30, len(X_treino)),
    random_state=SEMENTE_ALEATORIA,
)

metadados_bloco28 = {
    "versao_formato": 1,
    "modelo_principal": decisao_registrada_bloco28["modelo_principal"],
    "criterio_modelo_principal": decisao_registrada_bloco28["criterio"],
    "decisao_registrada_em_utc": decisao_registrada_bloco28[
        "registrado_em_utc"
    ],
    "versoes": {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scipy": version("scipy"),
        "scikit-learn": version("scikit-learn"),
        "joblib": version("joblib"),
    },
    "entradas": [
        {"nome": coluna, "tipo": str(tipo)}
        for coluna, tipo in X_treino.dtypes.items()
    ],
    "divisao_dados": {
        "arquivo_dados": "dataset.csv",
        "proporcao_teste_configurada": 0.30,
        "estratificada_pela_classe": True,
        "semente_aleatoria": SEMENTE_ALEATORIA,
        "registros_treino": len(X_treino),
        "registros_teste": len(X_teste),
    },
    "verificacao_salvamento": {
        "origem_amostra": "treino",
        "registros_amostra": len(amostra_treino_bloco28),
        "semente_aleatoria": SEMENTE_ALEATORIA,
    },
    "modelos": {},
}

pasta_modelos_bloco28.mkdir(exist_ok=True)

# Prepara e verifica todos os arquivos antes de colocá-los na pasta final.
with TemporaryDirectory(dir=pasta_modelos_bloco28) as nome_pasta_temporaria:
    pasta_temporaria_bloco28 = pasta_modelos_bloco28 / nome_pasta_temporaria
    arquivos_verificados_bloco28 = []

    for identificador, nome, pipeline in modelos_bloco27:
        arquivo_pipeline = pasta_temporaria_bloco28 / f"{identificador}.joblib"

        previsoes_originais = pipeline.predict(amostra_treino_bloco28)
        joblib.dump(pipeline, arquivo_pipeline, compress=3)

        pipeline_recarregado = joblib.load(arquivo_pipeline)
        previsoes_recarregadas = pipeline_recarregado.predict(
            amostra_treino_bloco28
        )

        if not np.array_equal(previsoes_originais, previsoes_recarregadas):
            raise ValueError(
                f"As previsões do pipeline recarregado diferem: {nome}."
            )

        metadados_bloco28["modelos"][identificador] = {
            "nome": nome,
            "arquivo": arquivo_pipeline.name,
            "classes": pipeline.named_steps["modelo"].classes_.tolist(),
            "parametros": decisao_registrada_bloco28["parametros_modelos"][
                identificador
            ],
        }
        arquivos_verificados_bloco28.append(arquivo_pipeline)
        print(f"Pipeline recarregado e verificado: {nome}")

    arquivo_metadados_bloco28 = pasta_temporaria_bloco28 / "metadados.json"

    with arquivo_metadados_bloco28.open("w", encoding="utf-8") as arquivo:
        json.dump(metadados_bloco28, arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")

    for arquivo_pipeline in arquivos_verificados_bloco28:
        arquivo_pipeline.replace(pasta_modelos_bloco28 / arquivo_pipeline.name)

    arquivo_metadados_bloco28.replace(pasta_modelos_bloco28 / "metadados.json")

print(f"\nPipelines salvos: {len(metadados_bloco28['modelos'])}")
print(f"Pasta dos modelos: {pasta_modelos_bloco28}")
print("Metadados salvos em: modelos/metadados.json")
print("Bloco 28 concluído.")



