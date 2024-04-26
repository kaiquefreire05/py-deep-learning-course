#%% Importações

from minisom import MiniSom
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot

#%% Carregando e dividindo base de dados

base = pd.read_csv('dataset-credit-data/credit_data.csv')

base = base.dropna()

base.loc[base.age < 0, 'age'] = 40.92

X = base.drop('default', axis= 1).values

y = base['default'].values

#%% Normalizando a base de dados

normalizer = MinMaxScaler(feature_range= (0, 1))

X = normalizer.fit_transform(X)

#%% Construção do mapa auto organizável

# Inicializa o modelo
som = MiniSom(x= 15, y= 15, input_len= 4, random_seed= 42)

# Inicializando pesos aleatóriamente
som.random_weights_init(X)

# Treinando o modelo com 100 iterações
som.train_random(data= X, num_iteration= 100)

#%% Plotagem do mapa de instâncias 

pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = colors[y[i]], markeredgewidth = 2)

#%% Detectando suspeitas de fraudes

# Criando um mapeamento entre os padrões de entrada e as unidades de saída da SOM
mapeamento = som.win_map(X)

# Concatenando as unidades vencedoras correspondentes aos padrões suspeitos
suspeitos = np.concatenate((mapeamento[(13,9)], mapeamento[(1,10)]), axis=0)

# Revertendo a normalização dos dados dos suspeitos para a escala original
suspeitos = normalizer.inverse_transform(suspeitos)

# Inicializando uma lista para armazenar as classes dos suspeitos
classe = []

# Iterando sobre cada linha na base de dados
for i in range(len(base)):
    # Iterando sobre cada suspeito
    for j in range(len(suspeitos)):
        # Verificando se o ID do suspeito corresponde ao ID na base de dados
        if base.iloc[i, 0] == int(round(suspeitos[j, 0])):
            # Se houver correspondência, adiciona a classe do suspeito à lista
            classe.append(base.iloc[i, 4])

# Convertendo a lista de classes para um array numpy
classe = np.asarray(classe)

# Combinando os dados dos suspeitos com suas respectivas classes
suspeitos_final = np.column_stack((suspeitos, classe))

# Ordenando os suspeitos finais com base nas classes
suspeitos_final = suspeitos_final[suspeitos_final[:, 4].argsort()]