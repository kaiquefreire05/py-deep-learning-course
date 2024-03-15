#%% Importações

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

#%% Carregando os dados

seed = 5
np.random.seed(seed)

(x, y), (x_test, y_test) = mnist.load_data() # Carregando as variáveis

# Reestruturando as imagens e passando para cinza
x = x.reshape(x.shape[0], 28, 28, 1) 

# Converte e normaliza os dados [0, 1]
x = x.astype('float32')
x /= 255

# Modelando para o formato OneHot e criando 10 colunas
y = to_categorical(y, 10) 

#%% Validação cruzada

kfold = StratifiedKFold(n_splits= 5, shuffle= True, random_state= seed)
resultados = []

for i_train, i_test in kfold.split(X= x, y= np.zeros(shape= (y.shape[0], 1))):
    
    # Criando Rede Neural Sequencial
    classificador = Sequential()
    
    # Adicionando camada de convolução para extrair as características
    classificador.add(Conv2D(filters= 32, kernel_size= (3, 3), input_shape= (28, 28, 1), activation= 'relu'))
    
    # Adicionando camada de Pooling para reduzir a dimensionalidade
    classificador.add(MaxPooling2D(pool_size= (2, 2)))
    
    # Transformando as imagens em arrays
    classificador.add(Flatten())
    
    # Camada oculta da Rede Neural Densa
    classificador.add(Dense(units= 128, activation= 'relu'))
    
    # Camada de saída
    classificador.add(Dense(units= 10, activation= 'softmax'))
    
    # Compilando Rede Neural
    classificador.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics= ['accuracy'])
    
    # Fazendo treinamento com uma parte da base de dados
    classificador.fit(x[i_train], y[i_train], batch_size= 128, epochs= 5)
    
    # Fazendo score com outra parte da base de dados
    precision = classificador.evaluate(x[i_test], y[i_test])
    
    resultados.append(precision[1])

media = sum(resultados) / len(resultados)
print(f'A média dos resultados é: {media}')