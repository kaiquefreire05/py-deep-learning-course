#%% Importações

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from matplotlib.pylab import plot
from matplotlib.pylab import pcolor
from matplotlib.pylab import colorbar

#%% Carregando dados e normalizando

base1 = pd.read_csv('dataset-cancer/entradas_breast.csv')
X = base1.iloc[:, 0:30].values
base2 = pd.read_csv('dataset-cancer/saidas_breast.csv')
y = base2.iloc[:,0].values

normalizer = MinMaxScaler(feature_range= (0, 1))

X = normalizer.fit_transform(X)

#%% Criando o mapa auto organizável

# Inicializa o modelo
som = MiniSom(x= 11, y= 11, input_len= 30, sigma= 1.0, learning_rate= 0.5
              , random_seed= 42)
som.random_weights_init(X) # Inicializando os pesos aleatóriamente
som.train(data= X, num_iteration= 10) # Treinando o modelo
          
pesos_som = som._weights # Retorna os pesos do mapa
activation_map = som._activation_map # Retorna o mapa de ativação

activation_res = som.activation_response(X) # Resposta de ativação

#%%  Plotagem do mapa de distâncias da SOM
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    
    w = som.winner(x)  # Encontra o neurônio vencedor para cada entrada
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = colors[y[i]], markeredgewidth = 2)
