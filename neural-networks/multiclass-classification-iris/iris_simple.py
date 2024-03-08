#%% Importações

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

#%% Carregando os dados

df = pd.read_csv('dataset/iris.csv')

#%% Dividindo entre previsores e classes

x = df.drop('class', axis= 1).values
y = df['class'].values

#%% Transformando as classes

encoder = LabelEncoder()
y = encoder.fit_transform(y)
y_dummy = to_categorical(y)

#%% Dividindo entre base de treino e teste

x_train, x_test, y_train, y_test = train_test_split(x, y_dummy, test_size= 0.25, random_state= 42)

#%% Contrução e treinamento da Rede Neural

""" 
    Método para encotrar quantidade de neutros na entrada é (número de colunas + números de classes) / 2
    Neste caso: (4 + 3) / 2
"""
classificador = Sequential() # instânciando Rede Neural Sequencial
classificador.add(Dense(units= 4, activation= 'relu', input_dim= 4)) #  camada de entrada
classificador.add(Dense(units= 4, activation= 'relu')) # camda escondida
classificador.add(Dense(units= 3, activation= 'softmax')) # camada de saída
classificador.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics= ['categorical_accuracy']) # compilando Rede Neural
classificador.fit(x= x_train, y= y_train, batch_size= 10, epochs= 1000)

#%% Fazendo previsões

resultado = classificador.evaluate(x= x_test, y= y_test)

print(f'A Rede Neural teve {resultado[0]} de perda.')
print(f'A Rede Neura teve {resultado[1]} de acerto.')

previsoes = classificador.predict(x_test)
previsoes = (previsoes > 0.5)

y_test_confuse = [np.argmax(t) for t in y_test]
previsoes_confuse = [np.argmax(t) for t in previsoes]
confuse_rn = confusion_matrix(previsoes_confuse, y_test_confuse)

