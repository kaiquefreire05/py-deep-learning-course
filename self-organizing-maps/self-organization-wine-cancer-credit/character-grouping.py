#%% Importações

from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from matplotlib.pyplot import pcolor
from matplotlib.pyplot import plot
from matplotlib.pyplot import colorbar

#%% Carregamento e pre-processamento

base = pd.read_csv('dataset-characters/personagens.csv')

X = base.drop('classe', axis= 1).values
y = base['classe'].values

# bart = 0, homer = 1
y[y == 'Bart'] = 0
y[y == 'Homer'] = 1

normalizer = MinMaxScaler(feature_range= (0, 1))

X = normalizer.fit_transform(X)

# Criando o SOM

som = MiniSom(x= 9, y= 9, input_len= 6, random_seed= 42)

# Inicializando os pesos de forma aleatória
som.random_weights_init(X)

# Treinando o mapa
som.train_random(data= X, num_iteration= 500, verbose= True)

#%% Plots do mapa

pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = colors[y[i]], markeredgewidth = 2)
    
#%% Detectando as classes

# Pegando o mapa ganhador

mapeamento = som.win_map(X)
suspeitos = mapeamento[(6,4)]
suspeitos = normalizer.inverse_transform(suspeitos)

classes = list()

for i in range(len(base)):
    for j in range(len(suspeitos)):
        if ((base.iloc[i, 0] == suspeitos[j, 0]) and 
            (base.iloc[i, 1] == suspeitos[j, 1]) and
            (base.iloc[i, 2] == suspeitos[j, 2]) and
            (base.iloc[i, 3] == suspeitos[j, 3]) and
            (base.iloc[i, 4] == suspeitos[j, 4]) and
            (base.iloc[i, 5] == suspeitos[j, 5])):
            classes.append(base.iloc[i, 6])
            
classes = np.asarray(classes)
final = np.column_stack((suspeitos, classes))
final = final[final[:, 4].args]
            