#%% Importações

import pandas as pd
import numpy as np
from keras.models import model_from_json

#%% Carregando a rede neural

arquivo = open('modelos-nn-salvos/classificador_breast.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)

#%% Carregando os pesos da rede neural

classificador.load_weights('modelos-nn-salvos/classificador_breast.h5')

#%% testando uma previsão 

novo_predict = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                          0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                          0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                          0.84, 158, 0.363]])

previsao = classificador.predict(novo_predict)
previsao = (previsao > 0.5)

x = pd.read_csv('datasets/entradas_breast.csv')
y = pd.read_csv('datasets/saidas_breast.csv')

classificador.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics= ['binary_accuracy'])

resultados = classificador.evaluate(x, y)