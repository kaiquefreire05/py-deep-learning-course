#%% Importações

import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

#%% Importando as variáveis

with open ('variaveis-processadas/autos.pkl', 'rb') as f:
    x, y = pickle.load(f)
    
#%% Função para criação da Rede Neural

def create_neural_regressor(loss):
    
    regr = Sequential()
    regr.add(Dense(units= 158, activation= 'relu', input_dim= 316)) # camada de entrada
    regr.add(Dropout(0.1))
    regr.add(Dense(units= 158, activation= 'relu')) # camada oculta
    regr.add(Dropout(0.1))
    regr.add(Dense(units= 1, activation= 'linear'))
    regr.compile(optimizer= 'adam', loss= loss, metrics= ['mean_absolute_error'])
    
    return regr

#%% Criando Rede Neural e fazendo o GridSearch

regressor = KerasRegressor(model= create_neural_regressor, epochs= 100, batch_size= 300)

parametros = {
    'model__loss': ['mean_absolute_error', 'mean_squared_error', 'squared_hinge']
    }
    
teste_regressor = GridSearchCV(estimator= regressor, param_grid= parametros, cv= 10)
teste_regressor = teste_regressor.fit(X= x, y= y)

best_loss = teste_regressor.best_params_
best_score = teste_regressor.best_score_

"""
    Teste 1:
        mean_absolute_error
        0.6150750587145749
"""