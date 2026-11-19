from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


RESULTADOS = Path(__file__).resolve().parent / "resultados"
METRICAS = {
    "F1 macro": ("F1 macro teste", "Combina precisão e recall por classe. A média dá o mesmo peso a cada classe."),
    "Acurácia": ("Acurácia teste", "Percentual de previsões corretas entre todos os registros do teste."),
    "Precisão macro": ("Precisão macro teste", "Para cada classe, mede quantas previsões daquela classe estavam corretas. A média dá o mesmo peso às três classes."),
    "Recall macro": ("Recall macro teste", "Para cada classe, mede quantos casos reais daquela classe foram identificados. A média dá o mesmo peso às três classes."),
}
CLASSES = {"Dropout": "Evasão", "Enrolled": "Matriculado", "Graduate": "Formado"}


def ler_csv(nome, **opcoes):
    return pd.read_csv(
        RESULTADOS / nome, sep=";", decimal=",", encoding="utf-8-sig", **opcoes
    )


def percentual(valor):
    return f"{valor:.2%}".replace(".", ",")


st.set_page_config(page_title="Desafio do Classificador Mestre", layout="wide")
st.title("Desafio do Classificador Mestre")
st.caption("Aprendizagem de Máquina · FATEC Indaiatuba · Grupo 2")
st.markdown("**Evasão e sucesso acadêmico** · Avaliação no conjunto de teste")

try:
    resultados = ler_csv("metricas_gerais_teste.csv")
    matriz = ler_csv("logistica_matriz_confusao_teste.csv", index_col=0)
    principal = resultados.loc[resultados["Papel"].eq("Principal")].iloc[0]
    matriz = matriz.loc[list(CLASSES), list(CLASSES)].rename(
        index=CLASSES, columns=CLASSES
    )
    resultados[[coluna for coluna, _ in METRICAS.values()]] = resultados[
        [coluna for coluna, _ in METRICAS.values()]
    ].apply(pd.to_numeric)
except (OSError, ValueError, KeyError, IndexError) as erro:
    st.error(f"Confira os CSVs originais na pasta resultados. Detalhe: {erro}")
    st.stop()

indicadores = st.columns(3)
indicadores[0].metric("Modelo principal", principal["Modelo"])
indicadores[1].metric("F1 macro no teste", percentual(principal["F1 macro teste"]))
indicadores[2].metric("Registros no teste", f"{int(principal['Registros']):,}".replace(",", "."))

comparacao, erros, evidencia, figuras, metodologia = st.tabs(
    ["Comparação de modelos", "Acertos e erros", "Escolha do modelo", "Figuras do projeto", "Metodologia"]
)

with comparacao:
    st.subheader("Desempenho dos sete classificadores")
    nome_metrica = st.selectbox("Escolha a métrica", list(METRICAS))
    coluna, definicao = METRICAS[nome_metrica]
    st.caption(definicao)
    ranking = resultados.sort_values(coluna, ascending=False)
    grafico = px.bar(
        ranking, x=coluna, y="Modelo", color="Papel", orientation="h",
        text=coluna,
        color_discrete_map={"Principal": "#007D78", "Comparação": "#456487"},
        category_orders={"Modelo": ranking["Modelo"].tolist()},
        labels={coluna: nome_metrica},
    )
    grafico.update_traces(texttemplate="%{x:.2%}", textposition="outside")
    grafico.update_xaxes(tickformat=".0%", range=[0, 1.09], tickvals=[0, .25, .5, .75, 1])
    grafico.update_yaxes(autorange="reversed", title=None)
    grafico.update_layout(legend_title_text=None, height=400)
    st.plotly_chart(grafico)
    lider = ranking.iloc[0]
    st.markdown(
        f"**Como ler.** Barras mais longas indicam valores maiores. "
        f"Nesta métrica, **{lider['Modelo']}** teve o maior valor observado "
        f"no teste ({percentual(lider[coluna])}). A cor verde identifica "
        f"o modelo principal, escolhido pela validação cruzada."
    )
    st.caption("Diferenças observadas no teste não demonstram superioridade estatística.")
    with st.expander("Ver tabela completa"):
        tabela = ranking[["Modelo", "Papel", *[c for c, _ in METRICAS.values()]]]
        st.dataframe(tabela.style.format({c: percentual for c, _ in METRICAS.values()}), hide_index=True)

with erros:
    st.subheader(f"Matriz de confusão · {principal['Modelo']}")
    grafico_matriz = px.imshow(
        matriz, text_auto=True, color_continuous_scale="Blues", aspect="auto",
        labels={"x": "Classe prevista", "y": "Classe real", "color": "Registros"},
    )
    grafico_matriz.update_layout(height=390)
    st.plotly_chart(grafico_matriz)
    evasao = matriz.loc["Evasão"]
    st.markdown(
        "**Como ler.** Cada linha corresponde ao resultado real do aluno. "
        "Cada coluna mostra a previsão do modelo. Os números na diagonal são acertos."
    )
    st.markdown(
        f"**Exemplo.** Dos **{int(evasao.sum())}** casos reais de evasão, "
        f"o modelo identificou corretamente **{int(evasao['Evasão'])}**. "
        f"Os demais foram previstos como matriculado (**{int(evasao['Matriculado'])}**) "
        f"ou formado (**{int(evasao['Formado'])}**)."
    )

with metodologia:
    st.subheader("Origem e período dos dados")
    st.markdown(
        "**Instituição:** Instituto Politécnico de Portalegre.  \n"
        "**Localização:** Portalegre, Alto Alentejo, Portugal.  \n"
        "**Período das matrículas:** do ano letivo de 2008/2009 ao de 2018/2019.  \n"
        "**Base analisada:** 4.424 registros de estudantes."
    )
    st.caption(
        "Os resultados descrevem a base dessa instituição. "
        "A aplicação a outras instituições ou períodos exige nova validação."
    )
    st.markdown(
        "[Artigo que descreve a base (2022)](https://doi.org/10.3390/data7110146) · "
        "[Referência para o período das matrículas](https://www.mdpi.com/2227-7390/14/4/626)"
    )

    st.subheader("Qualidade dos dados")
    st.markdown(
        "- **Valores nulos:** 0 células com valores ausentes, em 0 de 4.424 registros.\n"
        "- **Registros duplicados:** 0 linhas repetidas integralmente, considerando todas as colunas."
    )

    st.subheader("Como os modelos foram avaliados")
    st.markdown(
        "- O conjunto de 4.424 registros foi dividido aleatoriamente em "
        "**3.096 para treino (70%)** e **1.328 para teste (30%)**.\n"
        "- A divisão foi estratificada pela classe, mantendo aproximadamente "
        "a proporção de alunos em cada categoria. A semente fixa "
        "`random_state=42` permite reproduzir a divisão.\n"
        "- O teste final teve 427 casos de evasão, 238 alunos matriculados "
        "e 663 formados.\n"
        "- A Regressão Logística foi escolhida pelo maior F1 macro médio "
        "na validação cruzada de cinco divisões, antes da avaliação no teste.\n"
        "- As entradas incluem informações dos dois primeiros semestres. "
        "A análise representa esse cenário acadêmico."
    )

    st.subheader("Pré-processamento dos dados")
    st.markdown(
        "As variáveis numéricas foram padronizadas com `StandardScaler`, "
        "as categóricas foram codificadas com `OneHotEncoder` e as binárias "
        "foram mantidas como 0 ou 1. Na validação cruzada, o pré-processamento "
        "foi ajustado apenas com os dados de treino de cada divisão."
    )

with evidencia:
    st.header("Por que a Regressão Logística?")
    consolidado = ler_csv("comparacao_iniciais_ajustados_consolidada.csv")
    validacao = consolidado.loc[consolidado["Métrica"].eq("f1_macro")].copy()
    desempenho = "Média validação ajustado"
    tempo = "Tempo OOF total (s)"
    validacao = validacao.merge(ler_csv("tempos_validacao_oof.csv"), on="Modelo", validate="one_to_one")
    validacao = validacao.sort_values(desempenho, ascending=False)
    escolhido = validacao.loc[validacao["Modelo"].eq(principal["Modelo"])].iloc[0]
    segundo = validacao.iloc[1]
    rapido = validacao.loc[validacao[tempo].idxmin()]
    lider_teste = resultados.loc[resultados["F1 macro teste"].idxmax()]
    vantagem = (escolhido[desempenho] - segundo[desempenho]) * 100
    st.markdown(
        f"**Critério de escolha.** A Regressão Logística liderou o F1 macro médio na "
        f"validação cruzada ({percentual(escolhido[desempenho])}), "
        f"{vantagem:.2f} ponto percentual à frente de {segundo['Modelo']}. "
        f"O F1 macro dá o mesmo peso às três classes."
    )
    st.markdown(
        f"**Resultado no teste.** {lider_teste['Modelo']} obteve o maior F1 macro observado "
        f"({percentual(lider_teste['F1 macro teste'])}). A escolha foi feita antes do teste."
    )
    st.subheader("Tempo total da validação fora da dobra")
    grafico_tempo = px.bar(
        validacao.sort_values(tempo), x=tempo, y="Modelo", orientation="h",
        color_discrete_sequence=["#456487"], text_auto=".2f",
        labels={tempo: "Tempo total (segundos)"},
    )
    grafico_tempo.update_yaxes(autorange="reversed", title=None)
    st.plotly_chart(grafico_tempo)
    st.markdown(
        f"**Como ler.** Barras menores indicam menor tempo medido. "
        f"{rapido['Modelo']} foi o mais rápido ({rapido[tempo]:.2f} s). "
        f"A Regressão Logística levou {escolhido[tempo]:.2f} s nesta execução."
    )
    st.caption(
        "A medição reúne os ajustes e as previsões das cinco divisões. "
        "O tempo para prever um único aluno no dashboard ainda não foi medido."
    )
    with st.expander("Ver métricas da validação e tempos"):
        st.dataframe(validacao[["Modelo", desempenho, "DP validação ajustado", tempo]], hide_index=True)

    st.subheader("Simulação com mais registros")
    st.caption(
        "A referência é o tempo OOF medido com 3.096 registros de treino. "
        "A simulação mantém a divisão 70/30, o mesmo computador, as 34 entradas "
        "e as configurações dos modelos."
    )
    tamanho = st.selectbox(
        "Total de registros no cenário", [4424, 10000, 20000, 50000], index=1,
        format_func=lambda n: f"{n:,}".replace(",", "."),
    )
    treino_base = 3096
    treino_cenario = tamanho - (3 * tamanho + 9) // 10
    fator = treino_cenario / treino_base
    cenarios = validacao[["Modelo", tempo]].assign(
        **{
            "Crescimento linear (s)": validacao[tempo] * fator,
            "Crescimento quadrático (s)": validacao[tempo] * fator**2,
        }
    )
    st.dataframe(
        cenarios,
        hide_index=True,
        column_config={
            coluna: st.column_config.NumberColumn(format="%.2f")
            for coluna in cenarios.columns if coluna != "Modelo"
        },
    )
    st.markdown(
        f"**Como interpretar.** Este cenário teria **{treino_cenario:,}** registros de treino. "
        f"O cálculo multiplica o tempo medido por **{fator:.2f}** no cenário linear "
        f"e por **{fator**2:.2f}** no quadrático."
    )
    st.caption(
        "Os dois valores ilustram hipóteses matemáticas. Cada algoritmo pode crescer "
        "de forma diferente. Para estimar o tempo real de cada modelo, meça novamente "
        "com bases de tamanhos diferentes no mesmo computador."
    )


with figuras:
    st.header("Figuras do projeto")
    itens = [
        ("1 · Distribuição das classes", "figura_01_distribuicao_classes.png",
         "Compare as quantidades nas três classes. A turma de formados é a maior e a de matriculados é a menor. O F1 macro dá peso igual às classes apesar dessa diferença."),
        ("2 · Distribuições de idade e notas", "figura_02_distribuicoes_idade_notas.png",
         "Observe onde os valores se concentram e quais aparecem afastados da maioria. Valores extremos merecem investigação e foram mantidos na análise."),
        ("3 · Correlações de Spearman", "figura_03_correlacoes_spearman.png",
         "Leia o sinal para identificar a direção da associação e a distância de zero para avaliar sua intensidade. Uma associação entre variáveis, por si só, não estabelece causa."),
        ("4 · Comparação dos modelos iniciais", "figura_04_comparacao_modelos_iniciais.png",
         "Compare os modelos antes do ajuste. O classificador Dummy fornece uma referência simples para interpretar os resultados."),
        ("5 · Modelos antes e depois do ajuste", "figura_05_comparacao_iniciais_ajustados.png",
         "Compare as duas barras de cada modelo para ver a mudança média no F1 macro. As barras de erro mostram um desvio padrão entre as divisões."),
        ("6 · Matrizes na validação cruzada", "figura_06_matrizes_confusao_validacao_oof.png",
         "As linhas mostram a classe real e as colunas mostram a previsão. Leia os percentuais por linha. Examine a classe Matriculado, que teve o menor F1 nos sete modelos."),
        ("7 · Matrizes no conjunto de teste", "figura_07_matrizes_confusao_teste.png",
         "Compare as diagonais para ver a proporção de acertos dentro de cada classe real. Os demais valores mostram para quais classes os registros foram encaminhados pelo modelo."),
        ("8 · Desempenho acadêmico por classe no treino",
         "figura_08_desempenho_por_classe_treino.png",
         "No conjunto de treino, a mediana de aprovações no segundo semestre "
         "foi de 0 para evasão, 4 para matriculados e 6 para formados. "
         "As distribuições se sobrepõem entre as classes."),
    ]
    titulo = st.selectbox("Selecione uma figura", [titulo for titulo, _, _ in itens])
    _, nome_arquivo, comentario = next(item for item in itens if item[0] == titulo)
    st.subheader(titulo)
    st.markdown(f"**Como interpretar.** {comentario}")
    arquivo_figura = RESULTADOS / nome_arquivo
    if arquivo_figura.is_file():
        st.image(str(arquivo_figura))
    else:
        st.warning(f"Figura ausente em resultados: {nome_arquivo}")
