#%% Importações

import pickle
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import Model

#%% Importando as variáveis

with open('variáveis/variaveis_games', 'rb') as f:
    x, y_na, y_eu, y_jp = pickle.load(f)
    
#%% Estrutura da Rede Neural

camada_entrada = Input(shape=(61,))
oculta_one = Dense(units= 32, activation= 'sigmoid')(camada_entrada)
oculta_two = Dense(units= 32, activation= 'sigmoid')(oculta_one)
saida_one = Dense(units= 1, activation= 'linear')(oculta_two)
saida_two = Dense(units= 1, activation= 'linear')(oculta_two)
saida_three = Dense(units= 1, activation= 'linear')(oculta_two)

regressor = Model(inputs= camada_entrada,
                  outputs= [saida_one, saida_two, saida_three])
regressor.compile(optimizer= 'adam', loss= 'mse')
regressor.fit(x, [y_na, y_eu, y_jp],
              epochs= 5000, batch_size= 100)

#%% Previsões

previsoes_na, previsoes_eu, previsoes_jp = regressor.predict(x)