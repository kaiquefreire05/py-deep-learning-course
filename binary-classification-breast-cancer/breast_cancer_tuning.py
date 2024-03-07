#$$ Importações

import keras
import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

#%% Lendo dataframes

x = pd.read_csv('datasets/entradas_breast.csv')
y = pd.read_csv('datasets/saidas_breast.csv')

#%% Criando class para criar classificador
    
def create_neural_network(optimizer, loss, kernel_initializer, activation, neurons):
    """
        Cria uma rede neural para classificação binária.

    Retorna:
        Sequential: Modelo de rede neural sequencial configurado.
    """
    
    # Instanciando modelo sequencial
    classificador = Sequential()
        
    # Criando camada de entrada com os parâmetros da função
    classificador.add(Dense(units= neurons, activation= activation, kernel_initializer= kernel_initializer, input_dim= 30))
        
    # Dropout para evitar overfiting 
    classificador.add(Dropout(0.2))
    
    # Criando camada oculta
    classificador.add(Dense(units= neurons, activation= activation, kernel_initializer= kernel_initializer))
        
    # Dropout para evitar overfiting
    classificador.add(Dropout(0.2))
        
    # Criando camada de saída
    classificador.add(Dense(units= 1, activation= 'sigmoid'))
        
    # Compilando Rede Neural
    classificador.compile(optimizer= optimizer, loss = loss, metrics= ['binary_accuracy'])
        
    return classificador
    
   
#%% Criando modelo sequencial

# Definição do classificador a ser analisado.
classificador = KerasClassifier(model= create_neural_network)
 
# Definição dos parâmetros a serem testados.
parametros = {
    'batch_size': [10, 30],
    'epochs': [50, 100, 200],
    'model__optimizer': ['adam', 'sgd'],
    'model__loss': ['binary_crossentropy', 'hinge'],
    'model__kernel_initializer': ['random_uniform', 'normal'],
    'model__activation': ['relu', 'tanh'],
    'model__neurons': [16, 8]
}

grid_search = GridSearchCV(estimator= classificador, param_grid= parametros, scoring= 'accuracy', cv= 5) # criando gridsearch
grid_search = grid_search.fit(x, y) # fazendo o treinamento
 
best_params = grid_search.best_params_ # obtendo os melhores parâmetros
best_precision = grid_search.best_score_ # obtendo o melhor score

# batch_size: 10, epochs: 100, activation: relu, kernel_initializer: normal, loss: binary_crossentropy, neurons: 16, optimizer: adam