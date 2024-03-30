#%% Importações

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Carregando dados

base = pd.read_csv('dataset/poluicao.csv')

# Removendo os valores nulos
base = base.dropna()

# Dropando colunas indesejadas
base = base.drop(['No', 'year', 'month', 'day', 'hour', 'cbwd'], axis= 1)

# Dividindoo dataframe

train_base = base.drop('pm2.5', axis= 1).values
poluicao = base['pm2.5'].values

# Fazendo a normalização dos dados
normalizador = MinMaxScaler(feature_range= (0, 1))
train_base_normalized = normalizador.fit_transform(train_base)
poluicao = poluicao.reshape(-1, 1)
poluicao_normalized = normalizador.fit_transform(poluicao)

#%% Criação da estrutura de dados que representa a série temporal
# 10 horas anteriores para prever a hora atual

previsores = list()
poluicao_real = list()
for i in range(10, 41757):
    previsores.append(train_base_normalized[i-10:i, 0:6])
    poluicao_real.append(poluicao_normalized[i, 0])
    
previsores, poluicao_real = np.array(previsores), np.array(poluicao_real)

#%% Estrutura da Rede Neural Recorrente

regressor = Sequential()

regressor.add(LSTM(units= 100, return_sequences= True, input_shape= (previsores.shape[1], 6)))
regressor.add(Dropout(rate= 0.3))

regressor.add(LSTM(units= 50, return_sequences= True))
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

# EarlyStopping: interrompe o treinamento se a perda (loss) não melhorar após 10 épocas consecutivas
es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=1)

# ReduceLROnPlateau: reduz a taxa de aprendizado se a perda não melhorar após 5 épocas consecutivas
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)

# ModelCheckpoint: salva os pesos do modelo após cada época se houver uma melhoria na perda
mcp = ModelCheckpoint(filepath='pesos_modelo.h5', monitor='loss', save_best_only=True)

regressor.fit(x= previsores, y= poluicao_real, epochs= 100
              , batch_size= 64, callbacks= [es, rlr, mcp])

#%% Fazendo previsões

previsoes = regressor.predict(previsores)
previsoes = normalizador.inverse_transform(previsoes)

print(previsoes.mean())
print(poluicao.mean())

plt.figure(figsize= (10, 6))
plt.plot(poluicao, color= 'red', label= 'Poluição Real')
plt.plot(previsoes, color= 'blue', label= 'Previsão')
plt.legend()
plt.title('Gráfico de Previsão da Poluição', fontsize= 14, fontweight= 'bold')
plt.xlabel('Horas', fontsize= 10, fontweight= 'bold')
plt.ylabel('Valor da Poluição', fontsize= 10, fontweight= 'bold')
plt.grid(True, linestyle= '--')
plt.show()
