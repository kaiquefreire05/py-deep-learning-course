#%% Importações

import pandas as pd
import keras 
from keras.models import Sequential # criar rede neural sequencial
from keras.layers import Dense # crir camada densa
from keras.layers import Dropout # evitar overfiting
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score

#%% Lendo dataframes

x = pd.read_csv('datasets/entradas_breast.csv') 
y = pd.read_csv('datasets/saidas_breast.csv')

#%% Função para criar Rede Neural

class Classificador(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classificador = self.create_neural_network()

    def create_neural_network(self):
        """
        Cria um modelo de rede neural para classificação binária.

        Retorna:
        - classificador (Sequential): Modelo de rede neural compilado.

        A arquitetura da rede neural consiste em:
        - Camada de entrada com 30 características de entrada.
        - Duas camadas ocultas com 16 unidades cada, ativadas pela função ReLU.
        - Camada de saída com 1 unidade ativada pela função sigmoid.

        O modelo é compilado usando o otimizador Adam com uma taxa de aprendizado de 0.001,
        decaimento de peso de 0.01 e valor de clipe de 0.5. A função de perda usada é
        binary_crossentropy, e binary_accuracy é usada como métrica.
        """
        # Instanciando rede neural sequencial
        classificador = Sequential() 

        # Criando camada de entrada
        classificador.add(Dense(units=16, activation='relu', 
                                kernel_initializer='random_uniform', input_dim=30))
        
        classificador.add(Dropout(0.2)) # usando dropout para evitar overfiting
        
        # Criando camada oculta
        classificador.add(Dense(units=16, activation='relu', 
                                kernel_initializer='random_uniform'))
        
        classificador.add(Dropout(0.2)) # usando dropout para evitar overfiting

        # Criando camada de saída 
        classificador.add(Dense(units=1, activation='sigmoid'))

        # Criando otimizador
        otimizador = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.01, clipvalue=0.5)

        # Compilando Rede Neural
        classificador.compile(optimizer=otimizador, loss='binary_crossentropy', 
                              metrics=['binary_accuracy'])

        return classificador

    def fit(self, X, y):
        self.classificador.fit(X, y, epochs=100, batch_size=10)
        return self

    def predict(self, X):
        return (self.classificador.predict(X) > 0.5).astype(int)

#%% Criando classificador

classificador = Classificador()

#%% Obtendo resultados

results = cross_val_score(estimator= classificador, X= x, y= y, cv= 10, scoring= 'accuracy')

#%% Obtendo valores 

# media de accuracy da rede neural
media = results.mean()
print(f'A média dos resultados é: {media}')

# desvio

desvio = results.std()
print(f'O desvio dos resultados é de: {desvio}')