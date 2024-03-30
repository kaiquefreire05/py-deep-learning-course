#%% Importações 

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Carregando os dataframes

# Carregando as bases e dropando os valores nulos
base = pd.read_csv('datasets/petr4_treinamento.csv')
test = pd.read_csv('datasets/petr4_teste.csv')
base = base.dropna()
test = test.dropna()

# Divindo as colunas
train_base = base[['Open', 'High']].values
base_max_value = base['High'].values.reshape(-1, 1)

# Normalizando os Arrays
normalizer = MinMaxScaler(feature_range= (0, 1))
train_base_normalized = normalizer.fit_transform(train_base)
base_max_value_normalized = normalizer.fit_transform(base_max_value)

#%% Pegando os previsores e os preços reais 

previsores = list()  # Lista que guarda os 90 anteriores e o valore
real_price_one = list()  # Somente os resultados
real_price_two = list()

for i in range(90, 1242):
    previsores.append(train_base_normalized[i-90:i, 0])
    real_price_one.append(train_base_normalized[i, 0])
    real_price_two.append(train_base_normalized[i, 0])
    
# Transformando em um ArrayNumpy (formato que o Keras trabalha)
previsores = np.array(previsores)
real_price_one = np.array(real_price_one) 
real_price_two = np.array(real_price_two)
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

# Juntando os dois ArraysNumpy
preco_real = np.column_stack((real_price_one, real_price_two))

#%% Estrutura Rede Neural Recorrente

regressor = Sequential()

regressor.add(LSTM(units= 100, return_sequences= True, input_shape= (previsores.shape[1], 1)))
regressor.add(Dropout(rate= 0.3))

regressor.add(LSTM(units= 50, return_sequences= True))
regressor.add(Dropout(rate= 0.3))

regressor.add(LSTM(units= 50, return_sequences= True))
regressor.add(Dropout(rate= 0.3))

regressor.add(LSTM(units=50, return_sequences= False))
regressor.add(Dropout(rate= 0.3))

regressor.add(Dense(units= 2, activation= 'linear'))
regressor.compile(optimizer= 'rmsprop', loss= 'mean_squared_error', metrics= ['mean_absolute_error'])
regressor.fit(previsores, preco_real, epochs= 100, batch_size= 32)

#%% Prevendo base de teste

real_price_open = test['Open'].values
real_price_high = test['High'].values

# Pegando a base completa com os 90 dias anteriores
# Transformando a variável
# Normalizando os valores
base_completa = pd.concat((base['Open'], test['Open']), axis= 0)
entradas = base_completa[len(base_completa) - len(test) - 90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizer.transform(entradas)

x_test = list()

for i in range(90, 112):
    x_test.append(entradas[i-90:i, 0])

# Transformando para o padrão do Numpy
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

previsoes = regressor.predict(x_test)
previsoes = normalizer.inverse_transform(previsoes)

plt.figure(figsize= (10, 6))
plt.plot(real_price_open, color= 'blue', label= 'Preço Real High')
plt.plot(real_price_high, color= 'yellow', label= 'Preço Real High')
plt.plot(previsoes[:, 0], color= 'Red', label= 'Preço Previsões Abertura')
plt.plot(previsoes[:, 1], color= 'Orange', label= 'Preço Previsões Alta')
plt.legend()
plt.title('Comparação entre Preço Real e Previsões', fontsize=14, fontweight='bold')
plt.ylabel('Valor Yahoo', fontsize=10, fontweight='bold')
plt.xlabel('Tempo', fontsize=10, fontweight='bold')
plt.grid(True, linestyle='--')
plt.plot()