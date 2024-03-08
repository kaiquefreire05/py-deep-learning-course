#%% Importações

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from scikeras.wrappers import KerasClassifier

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import cross_val_score

#%% Carregando dataframe e fazendo pré-processamento

df = pd.read_csv('dataset/iris.csv') # lendo o dataset

x = df.drop('class', axis= 1).values # obtendo previssores
y = df['class'].values # obtendo as classes

encoder = LabelEncoder() # instânciando objeto LabelEncoder
y = encoder.fit_transform(y) # encodando as classes

y_dummy = to_categorical(y) # transformando as variáveis para categóricas

#%% Função para criar Rede Neural

def create_neural_network():
    
    classificador = Sequential() # instânciando rede sequential
    classificador.add(Dense(units= 4, activation= 'relu', input_dim= 4)) # camada de entrada
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units= 4, activation= 'relu')) # camada oculta
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units= 3, activation= 'softmax')) # camada de saída
    classificador.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics= ['categorical_accuracy']) # compilando
    
    return classificador

#%% Criando classificador

classificador = KerasClassifier(model= create_neural_network, epochs= 1000, batch_size= 10)

resultados = cross_val_score(estimator= classificador, X= x, y= y_dummy, cv= 10, scoring= 'accuracy')
media = resultados.mean()
desvio = resultados.std()

print(f'A média dos resultados da validação cruzada foi: {media}')
print(f'O desvio da validação cruzada foi: {desvio}')

"""
    Teste 1 (sem camadas de Dropout):
        Desvio: 0.3971
        Média: 0.7933
        
    Teste 2 (com camadas de Dropout):
        Desvio: 0.7733
        Média: 0.3335
"""
