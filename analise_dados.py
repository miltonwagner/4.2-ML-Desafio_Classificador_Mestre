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

