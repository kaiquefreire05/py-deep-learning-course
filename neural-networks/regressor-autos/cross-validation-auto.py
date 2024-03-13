#%% Importações

import pickle 

from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor

from sklearn.model_selection import cross_val_score

#%% Carregando as variáveis pré-processadas

with open('variaveis-processadas/autos.pkl', 'rb') as f:
    x, y = pickle.load(f)
    
#%% Função para criar a estrutura da Rede Neural

def create_neural_regressor():
    
    regressor = Sequential() # instânciando Rede Neural Sequencial
    regressor.add(Dense(units= 158, activation= 'relu', input_dim= 316)) # camada de entrada
    regressor.add(Dense(units= 158, activation= 'relu')) # camada oculta
    regressor.add(Dense(units= 1, activation= 'linear')) # camada de saída
    regressor.compile(optimizer= 'adam', loss= 'mean_absolute_error', metrics= ['mean_absolute_error']) # compilando Rede Neural
    
    return regressor

#%% Criando Rede Neural e fazendo treinamento

regressor = KerasRegressor(model= create_neural_regressor, epochs= 100, batch_size= 300)

results = cross_val_score(estimator= regressor, X= x, y=y, cv= 10, scoring= 'neg_mean_absolute_error')

media = results.mean()
desvio = results.std()