#%% Importações

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#%% Obtendo variáveis previsoes e de classe

x = pd.read_csv('datasets/entradas_breast.csv')
y = pd.read_csv('datasets/saidas_breast.csv')

#%% Criando e treinando Rede Neural com os melhores parâmetros

classificador = Sequential() #  instânciando rede neural
classificador.add(Dense(units= 16, activation= 'relu', kernel_initializer= 'normal', input_dim= 30)) # camada de entrada
classificador.add(Dropout(0.2)) # camada para evitar overfiting
classificador.add(Dense(units= 16, activation= 'relu', kernel_initializer= 'normal')) # camada oculta
classificador.add(Dense(units= 1, activation= 'sigmoid')) # camade de saída
classificador.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['binary_accuracy']) # compilando rede neural
classificador.fit(x= x, y= y, batch_size= 10, epochs= 100) # fazendo o treinamento da rede neural
 

#%% Prevendo um valor unitário

novo_predict = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                          0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                          0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                          0.84, 158, 0.363]])

previsao = classificador.predict(novo_predict)
previsao = (previsao > 0.5)