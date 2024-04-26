#%% Importações

from minisom import MiniSom  # Importa a classe MiniSom para criar e treinar a SOM (Self-Organizing Map)
import pandas as pd  # Importa a biblioteca pandas para manipulação de dados
from sklearn.preprocessing import MinMaxScaler  # Importa o MinMaxScaler para normalização de dados
from matplotlib.pylab import pcolor, colorbar, plot  # Importa funções de plotagem do matplotlib

#%% Carregando os dados

base = pd.read_csv('dataset-wine/wines.csv')  # Carrega os dados do arquivo CSV para um DataFrame
X = base.iloc[:,1:14].values  # Extrai as features do conjunto de dados
y = base.iloc[:,0].values  # Extrai os rótulos do conjunto de dados

#%% Processando e organizando os dados

normalizador = MinMaxScaler(feature_range = (0,1))  # Inicializa o normalizador MinMaxScaler
X = normalizador.fit_transform(X)  # Normaliza os dados de entrada

# Cria e inicializa uma SOM (Self-Organizing Map)
som = MiniSom(x = 8, y = 8, input_len = 13, sigma = 1.0, learning_rate = 0.5, random_seed = 2)
som.random_weights_init(X)  # Inicializa os pesos da SOM aleatoriamente
som.train_random(data = X, num_iteration = 100)  # Treina a SOM com os dados normalizados

som._weights  # Retorna os pesos finais da SOM
som._activation_map  # Retorna o mapa de ativação da SOM
q = som.activation_response(X)  # Calcula a resposta de ativação para cada entrada

# Plotagem do mapa de distâncias da SOM
pcolor(som.distance_map().T)
# MID - mean inter neuron distance
colorbar()

w = som.winner(X[2])  # Encontra o neurônio vencedor para a terceira entrada
markers = ['o', 's', 'D']  # Marcadores para cada classe
color = ['r', 'g', 'b']  # Cores para cada classe

# Mapeamento dos rótulos para 0, 1 e 2
y[y == 1] = 0
y[y == 2] = 1
y[y == 3] = 2

# Plotagem dos neurônios da SOM com os marcadores correspondentes às classes
for i, x in enumerate(X):
    w = som.winner(x)  # Encontra o neurônio vencedor para cada entrada
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],  # Plota o marcador na posição do neurônio vencedor
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = color[y[i]], markeredgewidth = 2)