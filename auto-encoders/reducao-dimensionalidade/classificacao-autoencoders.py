#%% Importações

import numpy as np
from keras.layers import Dense
from keras.layers import Input
from keras.models import Sequential
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical

#%% Carregamento da base de dados

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizando os valores para o padrão de 0 e 1
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Transformando as classes para o formato onehot
y_train_dummy = to_categorical(y_train)
y_test_dummy = to_categorical(y_test)

# Mudando o formato dos previsores
X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))

#%% Criando Rede Neural Sequencial (autoencoder)

autoencoder = Sequential()

# Camada de entrada e camada oculta
autoencoder.add(Dense(units= 32, activation= 'relu', input_dim= 784))

# Camada de saída
autoencoder.add(Dense(units= 784, activation='sigmoid'))

# Sumário da Rede Neural
autoencoder.summary()

# Compilando Rede Neural
autoencoder.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

# Treinando e validando os dados
autoencoder.fit(x= X_train, y= X_train, batch_size= 256
                , epochs= 100, validation_data= (X_test, X_test))

#%% Criando somente um encoder
dimensao_original = Input(shape= (784,))
camada_encoder = autoencoder.layers[0]
encoder = Model(dimensao_original, camada_encoder(dimensao_original))

X_train_codificados = encoder.predict(X_train)
X_test_codificados = encoder.predict(X_test)