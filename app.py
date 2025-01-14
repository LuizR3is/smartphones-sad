import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Carregar os dados
file_path = "smartphones.csv"  # Substitua pelo caminho correto do arquivo
df = pd.read_csv(file_path)

# Função para realizar o questionário
def realizar_questionario():
    print("Bem-vindo ao sistema de recomendação de smartphones!")

    # Pergunta 1: Definir o preço máximo
    preco_max = float(input("Qual o valor máximo que você deseja pagar em um smartphone? "))

    # Pergunta 2: Ranqueamento de atributos
    print("\nRanqueie os atributos a seguir de acordo com sua importância (1 = mais importante, 5 = menos importante):")
    atributos_visiveis = [
        "Desempenho (Velocidade do aparelho)",
        "Capacidade para abrir vários aplicativos",
        "Duração da bateria",
        "Espaço para guardar arquivos e aplicativos",
        "Qualidade das câmeras",
    ]
    atributos_internos = [
        "Veloc_Processador",
        "RAM",
        "Capac_Bateria",
        "Memória_Interna",
        "Câm_Tras_Principal",
    ]

    ranqueamento = {}
    for i, atributo in enumerate(atributos_visiveis):
        ranqueamento[atributos_internos[i]] = int(input(f"{atributo}: "))

    # Pergunta 3: Preferência de tamanho de tela
    tamanho_tela = input("\nQual sua preferência de tamanho de tela? (compacta, grande, indiferente): ").lower()

    # Pergunta 4: Marcas preferidas
    marcas_disponiveis = df["Marca"].unique()
    print("\nSelecione suas marcas preferidas (separe por vírgulas, ou digite 'todas' para considerar todas):")
    print(", ".join(marcas_disponiveis))
    marcas_escolhidas = input("Marcas preferidas: ").lower()
    if marcas_escolhidas == "todas":
        marcas_escolhidas = marcas_disponiveis
    else:
        marcas_escolhidas = [marca.strip() for marca in marcas_escolhidas.split(",")]

    user_preferences = {
        "Preco_Max": preco_max,
        "Preferencia_Caracteristicas": ranqueamento,
        "Tamanho_Tela": tamanho_tela,
        "Marcas": marcas_escolhidas
    }

    return user_preferences

# Função para calcular a regressão linear e avaliar os pesos das características
def calcular_pesos(df):
    # Selecionar características relevantes
    caracteristicas = ["Veloc_Processador", "RAM", "Memória_Interna", "Capac_Bateria", "Câm_Tras_Principal", "Câm_Front_Principal"]
    X = df[caracteristicas]
    y = df["Preço"]  # Usando o preço como variável dependente

    # Padronizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Treinar o modelo de regressão linear
    modelo = LinearRegression()
    modelo.fit(X_scaled, y)

    # Extrair os coeficientes como pesos das características
    pesos = dict(zip(caracteristicas, modelo.coef_))
    
    return pesos

# Aplicar o algoritmo de agrupamento
def realizar_agrupamento(df):
    features = ["Preço", "Veloc_Processador", "RAM"]
    df_selected = df[features].dropna()

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_selected)

    kmedoids = KMedoids(n_clusters=3, random_state=42, metric="euclidean")
    df["Cluster"] = kmedoids.fit_predict(df_scaled)

    cluster_labels = {0: "Aparelhos de Entrada", 1: "Aparelhos Intermediários", 2: "Aparelhos Premium"}
    df["Categoria"] = df["Cluster"].map(cluster_labels)
    return df

# Função para calcular o score baseado nas preferências do usuário
def calcular_score(smartphone, preferencias):
    score = 0
    peso = 5  # Peso inicial para a característica mais importante
    for caracteristica, ordem in preferencias.items():
        if caracteristica == "Veloc_Processador":
            score += peso * smartphone["Veloc_Processador"]
        elif caracteristica == "RAM":
            score += peso * smartphone["RAM"]
        elif caracteristica == "Capac_Bateria":
            score += peso * smartphone["Capac_Bateria"]
        elif caracteristica == "Memória_Interna":
            score += peso * smartphone["Memória_Interna"]
        elif caracteristica == "Câm_Tras_Principal":
            score += peso * (smartphone["Câm_Tras_Principal"] + smartphone["Câm_Front_Principal"])
        peso -= 1
    return score

# Treinar o modelo Random Forest
def treinar_random_forest(df, ranqueamento, user_preferences):
    df['Score'] = df.apply(lambda row: calcular_score(row, user_preferences["Preferencia_Caracteristicas"]), axis=1)

    atributos = list(ranqueamento.keys())
    df_treino = df[atributos + ["Score"]].dropna()

    X = df_treino[atributos]
    y = df_treino["Score"]

    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_rf.fit(X, y)

    return modelo_rf

# Revisão da função de recomendação
def recomendar_smartphones(df, user_preferences, ranqueamento, modelo_rf):
    preco_max = user_preferences["Preco_Max"]
    marcas_escolhidas = user_preferences["Marcas"]
    tamanho_tela = user_preferences["Tamanho_Tela"]

    # Aplicar filtros iniciais
    df_filtrado = df[df["Preço"] <= preco_max]
    df_filtrado = df_filtrado[df_filtrado["Marca"].isin(marcas_escolhidas)]

    # Filtrar por tamanho de tela
    if tamanho_tela == "compacta":
        df_filtrado = df_filtrado[df_filtrado["Tamanho_Tela"] < 6.5]
    elif tamanho_tela == "grande":
        df_filtrado = df_filtrado[df_filtrado["Tamanho_Tela"] >= 6.5]

    # Verificar se há smartphones após o filtro
    if df_filtrado.empty:
        print("Nenhum smartphone encontrado com os critérios selecionados.")
        return pd.DataFrame()

    # Normalizar os atributos
    scaler = MinMaxScaler()
    atributos = list(ranqueamento.keys())
    df_filtrado[atributos] = scaler.fit_transform(df_filtrado[atributos])

    # Ajustar a ponderação conforme preferências do usuário
    for atributo in atributos:
        df_filtrado[atributo] *= (6 - ranqueamento[atributo])  # Peso inversamente proporcional à ordem

    # Calcular scores com base nos pesos ajustados
    df_filtrado["Score"] = df_filtrado[atributos].sum(axis=1) - abs(df_filtrado["Preço"] - preco_max) / preco_max

    # Prever relevância com o modelo Random Forest
    X_filtrado = df_filtrado[atributos]
    df_filtrado["Predict"] = modelo_rf.predict(X_filtrado)
    df_filtrado["Score"] += df_filtrado["Predict"]

    # Ordenar o DataFrame pelo score (maior score = mais adequado)
    df_sorted = df_filtrado.sort_values(by="Score", ascending=False)

    # Selecionar os 3 smartphones com maior score
    top_3 = df_sorted.head(3)

    # Selecionar a melhor opção de uma marca não escolhida
    df_nao_escolhidas = df[(~df["Marca"].isin(marcas_escolhidas)) & (df["Preço"] <= preco_max)]
    if tamanho_tela == "compacta":
        df_nao_escolhidas = df_nao_escolhidas[df_nao_escolhidas["Tamanho_Tela"] < 6.5]
    elif tamanho_tela == "grande":
        df_nao_escolhidas = df_nao_escolhidas[df_nao_escolhidas["Tamanho_Tela"] >= 6.5]
    df_nao_escolhidas[atributos] = scaler.transform(df_nao_escolhidas[atributos])    
    for atributo in atributos:
        df_nao_escolhidas[atributo] *= (6 - ranqueamento[atributo])

    df_nao_escolhidas["Score"] = df_nao_escolhidas[atributos].sum(axis=1) - \
                                abs(df_nao_escolhidas["Preço"] - preco_max) / preco_max

    X_nao_escolhidas = df_nao_escolhidas[atributos]
    df_nao_escolhidas["Predict"] = modelo_rf.predict(X_nao_escolhidas)
    df_nao_escolhidas["Score"] += df_nao_escolhidas["Predict"]
    melhor_nao_escolhido = df_nao_escolhidas.sort_values(by="Score", ascending=False).head(1)

    # Combinar as recomendações
    recomendacoes = pd.concat([top_3, melhor_nao_escolhido])

    # Sumarização das Recomendações com valores originais
    recomendacoes_resumidas = recomendacoes[["Marca", "Modelo", "Preço", "Veloc_Processador", "RAM", "Capac_Bateria", "Memória_Interna", "Tamanho_Tela", "Câm_Tras_Principal", "Câm_Front_Principal"]]

    return recomendacoes_resumidas

# Função para sumarizar as recomendações
def sumarizar_recomendacoes(recomendacoes, user_preferences):
    print("\nDetalhes das recomendações:")
    for i, (index, row) in enumerate(recomendacoes.iterrows()):
        modelo = row["Modelo"]
        marca = row["Marca"]
        preco = row["Preço"]
        print(f"\n{i+1}- {marca} {modelo}, R$ {preco:.2f}")

        # Destaques baseados no ranqueamento
        for atributo, peso in sorted(user_preferences["Preferencia_Caracteristicas"].items(), key=lambda x: x[1]):
            valor = df.loc[index, atributo]  # Usar os valores originais
            print(f"  - {atributo.replace('_', ' ')}: {valor}")

# Função para criar tabela gráfica comparativa
def criar_tabela_grafica(recomendacoes, df_original):

    # Extrair informações relevantes
    dados = []
    modelos = []
    for index, row in recomendacoes.iterrows():
        modelo = df_original.loc[index, "Modelo"]
        modelos.append(modelo)  # Adicionar o modelo como cabeçalho
        dados.append([
            f"R$ {df_original.loc[index, 'Preço']:.2f}",
            f"{df_original.loc[index, 'Veloc_Processador']} GHz",
            f"{df_original.loc[index, 'RAM']} GB",
            f"{df_original.loc[index, 'Memória_Interna']} GB",
            f"{df_original.loc[index, 'Capac_Bateria']} mAh",
            f"{df_original.loc[index, 'Câm_Tras_Principal']} / {df_original.loc[index, 'Câm_Front_Principal']} MP",
        ])

    colunas = ["Preço", "Processamento", "RAM", "Memória Interna", "Bateria", "Câmeras"]
    
    # Gerar o gráfico com Matplotlib
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("tight")
    ax.axis("off")
    tabela = ax.table(cellText=dados, colLabels=colunas, rowLabels=modelos, loc="center", cellLoc="center")
    
    # Ajustar tamanho da fonte e largura da coluna
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(8)  # Reduzir tamanho da fonte
    tabela.auto_set_column_width(col=list(range(len(colunas))))
    
    # Adicionar espaçamento entre as colunas para melhor ajuste
    for key, cell in tabela.get_celld().items():
        cell.set_aa(0.5)

    plt.title("Comparação entre Smartphones Recomendados", fontsize=10, weight="bold")
    plt.tight_layout()  # Ajustar layout para evitar cortes
    plt.show()

# Main
def main():
    # Realizar o questionário
    user_preferences = realizar_questionario()

    # Realizar agrupamento
    df_agrupado = realizar_agrupamento(df)

    # Realizar regressão para ver os pesos das variáveis
    pesos_regressao = calcular_pesos(df)
    print(pesos_regressao)

    # Treinar modelo Random Forest
    modelo_rf = treinar_random_forest(df_agrupado, user_preferences["Preferencia_Caracteristicas"], user_preferences)

    # Recomendar smartphones
    recomendacoes = recomendar_smartphones(df_agrupado, user_preferences, user_preferences["Preferencia_Caracteristicas"], modelo_rf)

    # Exibir recomendações
    sumarizar_recomendacoes(recomendacoes, user_preferences)

    # Mostrar tabela gráfica comparativa
    criar_tabela_grafica(recomendacoes, df)

if __name__ == "__main__":
    main()