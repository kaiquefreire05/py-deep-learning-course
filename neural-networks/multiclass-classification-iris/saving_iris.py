#%% Importações

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

#%% Fazendo o carregamento do dataframe

df = pd.read_csv('dataset/iris.csv')

#%% Separando previsores e classe

x = df.drop('class', axis= 1)
y = df['class']

encoder = LabelEncoder()
y = encoder.fit_transform(y)
y_dummy = to_categorical(y)

#%% Criando modelo Sequential e fazendo seu treinamento

classificador = Sequential()
classificador.add(Dense(units= 8, activation= 'relu', kernel_initializer='uniform', input_dim= 4))
classificador.add(Dropout(0.1))
classificador.add(Dense(units= 8, activation= 'relu', kernel_initializer='uniform'))
classificador.add(Dropout(0.1))
classificador.add(Dense(units= 3, activation= 'softmax', kernel_initializer= 'uniform'))
classificador.compile(optimizer= 'sgd', loss= 'mse', metrics= ['categorical_crossentropy'])
classificador.fit(x= x, y= y_dummy, batch_size= 10, epochs= 1000)

#%% Salvando Rede Neural

classificador_iris_json = classificador.to_json()

with open('classificador_iris_json', 'w') as json_file:
    json_file.write(classificador_iris_json)
    
classificador.save_weights('pesos_iris.h5')

#%% Carregando Rede Neural salva

arquivo = open('modelos-salvos/classificador_iris_json')
estrutura_rede = arquivo.read()
arquivo.close()

classificador2 = model_from_json(estrutura_rede)
classificador2.load_weights('modelos-salvos/pesos_iris.h5')

#%% Criar e classificar novo registro

novo = np.array([[3.2, 4.5, 0.9, 1.1]])
previsao = classificador2.predict(novo)
previsao = (previsao > 0.5)
if previsao[0][0] == True and previsao[0][1] == False and previsao[0][2] == False:
    print('Iris setosa')
elif previsao[0][0] == False and previsao[0][1] == True and previsao[0][2] == False:
    print('Iris virginica')
elif previsao[0][0] == False and previsao[0][1] == False and previsao[0][2] == True:
    print('Iris versicolor')