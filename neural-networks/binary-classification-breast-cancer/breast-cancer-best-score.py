# Importações

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from scikeras.wrappers import KerasClassifier

from sklearn.model_selection import cross_val_score

#%% Carregando entradas e saídas

x = pd.read_csv('datasets/entradas_breast.csv')
y = pd.read_csv('datasets/saidas_breast.csv')

#%% Função para criar estrutura da Rede Neural

def create_neural_classifier():
    
    classifier = Sequential()
    classifier.add(Dense(units= 16, activation= 'relu', input_dim= 30))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units= 16, activation= 'relu'))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units= 1, activation= 'sigmoid'))
    classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['binary_accuracy'])
        
    return classifier

#%% Criando Rede Neural e fazer validação cruzada

classificador = KerasClassifier(model= create_neural_classifier, epochs= 200, batch_size= 25)

resultados = cross_val_score(estimator= classificador, X= x, y= y, cv= 10)

media = resultados.mean()
desvio = resultados.std()

print(f'A média dos resultados da validação cruzada foi: {media}')
print(f'O desvio da validação cruzada foi: {desvio}')

"""
    Teste 1: epochs= 150, batch_size= 20
        média: 0.8168
        desvio: 0.1236
        resultados: 0.894737, 0.649123, 0.894737, 0.894737, 
        0.877193, 0.894737, 0.614035, 0.929825, 0.894737, 0.625
        
    Teste 2: epochs= 200, batch_size= 20
        média: 0.8682
        desvio: 0.04083
        resultados: 0.859649, 0.842105, 0.824561, 0.912281, 
        0.947368, 0.807018, 0.842105, 0.859649, 0.894737, 0.892857
    
    Teste 3: epochs= 200, batch_size= 25
    média: 0.8419
    desvio: 0664
    resultados:
        0.666667, 0.789474, 0.877193, 0.894737, 0.859649, 0.894737
        , 0.859649, 0.859649, 0.824561, 0.892857

"""