"""
    Usando base de dados extraídos no site Yahoo Finanças;
    Dados das ações da Petrobras no ano de 2017 a 2018;
    Usando Rede Neurais Recorrentes;
    Previsão de apenas um valor.
"""

#%% Importações

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

#%% Pré-processamento

base = pd.read_csv('datasets/petr4_treinamento.csv')
test = pd.read_csv('datasets/petr4_teste.csv')

# Removendo valores nulos da base 
base = base.dropna()

# Atribuindo a variável somente a coluna 'open'
base_train = base['Open'].values.reshape(-1, 1)

# Normalizando os dados
normalizador = MinMaxScaler(feature_range= (0, 1))
base_train_normalized = normalizador.fit_transform(X= base_train)

#%% Estrutura da base para previsão temporal

previsores = list() # Lista para armazenar a sequência de previsores
preco_real = list() # Lista para armazenar os valores reais

# Loop sobre os dados para criar as sequências de previsores e os preços reais
for i in range(90, 1242):  # Começa a partir do 90º dia até o final dos dados disponíveis
    # Adiciona os últimos 90 dias como previsores
    previsores.append(base_train_normalized[i-90:i, 0])
    # Adiciona o preço real do próximo dia
    preco_real.append(base_train_normalized[i, 0])

# Converte as listas em matrizes NumPy para uso posterior no treinamento do modelo
previsores, preco_real = np.array(previsores), np.array(preco_real)

# Redimensionando a variável para o formato que o keras trabalha
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

#%% Estrutura da Rede Neural

"""
    Apenas coloque 'return_sequences= True', quando na estrutura de sua Rede
    Neural tiver outra camada de LSTM
    
    input_shape= (previsores.shape[1], 1)
    previsores.shape[90], remete a quantidade de entradas (colunas) que a base
    possui e o segundo termo a quantidade de previsores
"""
regressor = Sequential()
regressor.add(LSTM(units= 100, return_sequences= True, input_shape= (previsores.shape[1], 1)))
regressor.add(Dropout(rate= 0.3))

regressor.add(LSTM(units= 50, return_sequences= True))
regressor.add(Dropout(rate= 0.3))

regressor.add(LSTM(units= 50, return_sequences= True))
regressor.add(Dropout(rate= 0.3))

regressor.add(LSTM(units= 50, return_sequences= False))
regressor.add(Dropout(rate= 0.3))

regressor.add(Dense(units= 1, activation= 'linear'))
regressor.compile(optimizer= 'rmsprop', loss= 'mean_squared_error'
                  , metrics= ['mean_absolute_error'])
regressor.fit(previsores, preco_real, epochs= 100, batch_size= 32)

#%% Prevendo preços das ações

# Carregando dados
base_teste = pd.read_csv('datasets/petr4_teste.csv')

# Atribuindo a um array somente a coluna 'Open'
real_price_test = base_teste['Open'].values

# Concatenando a coluna de treino com a de teste
base_completa = pd.concat((base['Open'], base_teste['Open']), axis= 0)

# Selecionado os últimos 90 registros dos dados de treinamento e teste combinados
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = entradas.reshape(-1, 1)  # Transformando para o formato que o Numpy trabalha
entradas = normalizador.transform(entradas)  # Normalizando os dados

# Inicializamos uma lista vazia para armazenar as sequências de dados de teste
x_test = list()

# Loop sobre um intervalo de dias, começando em 90 até 111
# Isso nos permite construir sequências de 90 dias para cada ponto de teste
for i in range(90, 112):
    # Selecionamos uma janela de 90 dias, começando em i-90 e indo até i
    # A segunda dimensão [0] seleciona apenas a primeira coluna (preço das ações) dos dados normalizados
    window = entradas[i-90:i, 0]
    # Adicionamos a janela atual à lista de sequências de teste
    x_test.append(window)

# Convertendo a lista de sequências em um array numpy
x_test = np.array(x_test)

# Remodelando o array para que ele tenha três dimensões, como esperado pelo modelo
# A terceira dimensão (1) representa o número de features, que é 1 (preço das ações)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

previsoes = regressor.predict(x_test)
previsoes = normalizador.inverse_transform(previsoes)
previsoes.mean()
real_price_test.mean()

#%% Plotagem de gráfico

plt.plot(real_price_test, color= 'red', label= 'Preço real')
plt.plot(previsoes, color= 'blue', label= 'Previsões')
plt.xlabel('Tempo', fontsize= 10, fontweight= 'bold')
plt.ylabel('Valor Yahoo', fontsize= 10, fontweight= 'bold')
plt.grid(True, linestyle= '--')
plt.legend()
plt.show()