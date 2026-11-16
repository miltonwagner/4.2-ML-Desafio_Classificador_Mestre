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

