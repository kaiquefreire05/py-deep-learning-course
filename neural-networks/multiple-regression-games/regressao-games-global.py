#%% Importações

import pickle
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Input

#%% Carregando variável previsoras e classe

with open('variáveis/games_global.pkl', 'rb') as f:
    x, y = pickle.load(f)
x = x.astype('float32')
#%% Estrutura da Rede Neural e treinamento

ativacao_ocultas = Activation(activation= 'sigmoid') # função de ativação
ativacao_saida = Activation(activation= 'linear')

camada_entrada = Input(shape= (65, )) # camada de entrada
oculta_one = Dense(units= 33, activation= ativacao_ocultas)(camada_entrada)
layer_dropout = Dropout(rate= 0.2)(oculta_one)
oculta_two = Dense(units= 33, activation= ativacao_ocultas)(layer_dropout)
layer_dropout_two = Dropout(rate= 0.2)(oculta_two)
saida = Dense(units= 1, activation= ativacao_saida)(layer_dropout_two)

regressor = Model(inputs= camada_entrada, outputs= saida)
regressor.compile(optimizer= 'adam', loss= 'mse')
regressor.fit(x= x, y= y, epochs= 5000, batch_size= 100)

#%% Previsões

previsoes_global = regressor.predict(x)
