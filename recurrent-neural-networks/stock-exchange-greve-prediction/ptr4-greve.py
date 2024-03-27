"""
    Usando uma base de dados extraídos do site Yahoo Finanças
    Fazer a predição dos valores usando Redes Neurais Recorrentes
    Nota: Os dados são da época da greve dos caminhoneiros
    Fazer análise em relação ao outro projeto
"""

#%% Importações

import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

#%% Carregando datasets

base = pd.read_csv('datasets/petr4_treinamento_ex.csv')
test = pd.read_csv('datasets/petr4_teste_ex.csv')

#%% Pre-processamento da variável de treino

# Dropando os valores nulos do dataset
base = base.dropna()

# Obtendo somente a coluna alvo do dataset
base_train = base['Open'].values.reshape(-1, 1)

# Criando normalizador 
normalizador = MinMaxScaler(feature_range= (0, 1))
base_train_normalized = normalizador.fit_transform(base_train)

# Previsores e classe com lookback de 90 dias

previsores = list()
preco_real = list()

for i in range(90, 1342):
    # Adiciona os últimos 90 dias como previsores
    previsores.append(base_train_normalized[i-90:i, 0])
    # Adicionando o preço real na lista
    preco_real.append(base_train_normalized[i, 0])
    
previsores = np.array(previsores)
preco_real = np.array(preco_real)

previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

#%% Estrutura da Rede Neural Recorrente

regressor = Sequential()
regressor.add(LSTM(units= 100, return_sequences= True, input_shape= (previsores.shape[1], 1)))
regressor.add(Dropout(rate= 0.2))

regressor.add(LSTM(units= 50, return_sequences= True))
regressor.add(Dropout(rate= 0.2))

regressor.add(LSTM(units= 50, return_sequences= True))
regressor.add(Dropout(rate= 0.2))

regressor.add(LSTM(units= 50, return_sequences= True))
regressor.add(Dropout(rate= 0.2))

regressor.add(LSTM(units= 50, return_sequences= False))
regressor.add(Dropout(rate= 0.2))

regressor.add(Dense(units= 1, activation= 'linear'))
regressor.compile(optimizer= 'rmsprop', loss= 'mean_squared_error'
                  , metrics= ['mean_absolute_error'])

regressor.fit(previsores, preco_real, epochs= 100, batch_size= 32)

#%% Preparando valores que vão ser previstos

# Array contendo os valores reais
real_price_test = test['Open'].values

# Concatenando os dois dataframes
complete_base = pd.concat((base['Open'], test['Open']), axis= 0) 

# Selecionando os últimos 90 registros da base e concatenando com os valores de teste
entradas = complete_base[len(complete_base) - len(test) - 90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

# Armazenar as sequências de dados de teste
x_test = list()

for i in range(90, 109):
    # Selecionamos uma janela de 90 dias, começando em i-90 e indo até i
    # A segunda dimensão [0] seleciona apenas a primeira coluna (preço das ações) dos dados normalizados
    window = entradas[i-90:i, 0]
    x_test.append(window)
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#%% Prevendo os valores

previsao = regressor.predict(x_test)
previsao = normalizador.inverse_transform(previsao)
print(previsao.mean())
print(real_price_test.mean())

# Plotando os dados
plt.figure(figsize=(10, 6))
plt.plot(real_price_test, color='red', label='Preço Real', linestyle='-')
plt.plot(previsao, color='blue', label='Previsões', linestyle='--')

# Destacando pontos de divergência
for i in range(len(real_price_test)):
    if real_price_test[i] != previsao[i]:
        plt.scatter(i, real_price_test[i], color='red', marker='x', s=50)
        plt.scatter(i, previsao[i], color='blue', marker='x', s=50)

# Adicionando título e rótulos dos eixos
plt.title('Comparação entre Preço Real e Previsões', fontsize=14, fontweight='bold')
plt.ylabel('Valor Yahoo', fontsize=10, fontweight='bold')
plt.xlabel('Tempo', fontsize=10, fontweight='bold')

# Adicionando legenda e grade
plt.legend()
plt.grid(True, linestyle='--')

# Exibindo o gráfico
plt.show()

