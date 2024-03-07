#%% importações

import pandas as pd
import numpy as np
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential

#%% Carregamento de previssores e classe

x = pd.read_csv('datasets/entradas_breast.csv')
y = pd.read_csv('datasets/saidas_breast.csv')

#%% Criando e treinando Rede Neural com os melhores parâmetros

classificador = Sequential() # instânciando Rede Neural Sequencial
classificador.add(Dense(units= 16, activation= 'relu', kernel_initializer= 'normal', input_dim= 30)) # criando camada de entrada
classificador.add(Dropout(0.2)) # camada para evitar overfiting
classificador.add(Dense(units= 16, activation= 'relu', kernel_initializer= 'normal')) # camada oculta
classificador.add(Dropout(0.2)) # camada para evitar overfiting
classificador.add(Dense(units= 1, activation= 'sigmoid')) # camada de saída usando função 'sigmoid'
classificador.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['binary_accuracy']) # compilando Rede Neural
classificador.fit(x= x, y= y, batch_size= 10, epochs= 100)

#%% Salvando Rede Neural

classificador_json = classificador.to_json()

with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)

# armazenando os pesos da Rede Neural
classificador.save_weights('classificador_breast.h5')