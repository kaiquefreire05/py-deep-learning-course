#%% Importações

import pandas as pd

from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from keras.layers import Dense
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

#%% Fazendo leitura e pré-processamento do dataframe

df= pd.read_csv('dataset/iris.csv') # lendo dataset

x = df.drop('class', axis= 1).values # obtendo previsores
y = df['class'].values # obtendo classe

encoder = LabelEncoder() # instânciando objeto LabelEncoder
y = encoder.fit_transform(y) # aplicando encoder nas classes
y_dummy = to_categorical(y) # aplicando 'OneHot'

#%% Função de criação da Rede Neural

def create_neural_network(units, activation, initializer, optimizer, loss):
    
    classificador = Sequential()
    classificador.add(Dense(units= units, activation= activation, kernel_initializer= initializer, input_dim= 4)) # camada de entrada
    classificador.add(Dense(units= units, activation= activation, kernel_initializer= initializer)) # camada oculta
    classificador.add(Dense(units= 3, activation= 'softmax', kernel_initializer= initializer)) # camada de saída
    classificador.compile(optimizer= optimizer, loss= loss, metrics= ['categorical_accuracy']) # compilando Rede Neural
    
    return classificador

#%% Testando GridSearch

classificador = KerasClassifier(model= create_neural_network)

param_grid = {
    "batch_size": [10, 30],
    "epochs": [500, 1000],
    "model__optimizer": ["adam", "sgd"],
    "model__loss": ["categorical_crossentropy", "mse"],
    "model__initializer": ["uniform", "normal"],
    "model__activation": ["relu", "tanh"],
    "model__units": [4, 8, 16]
    }

teste_grid = GridSearchCV(estimator= classificador, param_grid= param_grid, scoring= 'accuracy', cv= 5)
teste_grid.fit(x, y)

best_params = teste_grid.best_params_
best_score = teste_grid.best_score_

print(f'Os melhores parâmetros foram: {best_params}')
print(f'O melhor score foi: {best_score}')

"""
    Teste 1:
        Os melhores parâmetros foram: {'batch_size': 10, 'epochs': 1000, 'model__activation': 'relu', 'model__initializer': 'uniform', 'model__loss': 'mse', 'model__optimizer': 'sgd', 'model__units': 8}
        O melhor score foi: 0.5733333333333333
"""