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

#%% Rede Neural sem redução de dimensionalidade

classifier1 = Sequential()

# Entrada entrada e primeira camada oculta
# 784 entradas + 10 saídas / 2
classifier1.add(Dense(units= 397, activation= 'relu', input_dim= 784))

# Segunda camada oculta
classifier1.add(Dense(units= 394, activation= 'relu'))

# Camada de saída
classifier1.add(Dense(units= 10, activation= 'softmax'))

# Compilando Rede Neural
classifier1.compile(optimizer= 'adam', loss= 'categorical_crossentropy'
                    , metrics= ['accuracy'])

# Treinamento e validação
classifier1.fit(x= X_train, y= y_train_dummy, batch_size= 256
                , epochs= 100, validation_data= (X_test, y_test_dummy))

#%% Rede Neural com redução de dimensionalidade

# Criando RN Sequencial
classifier2 = Sequential()

# Adicionando primeira camada oculta e camada de saída
# 32 + 10 / 2
classifier2.add(Dense(units= 21, activation= 'relu', input_dim= 32))

# Segunda camada oculta
classifier2.add(Dense(units= 21, activation= 'relu'))

# Camada de saída
classifier2.add(Dense(units= 10, activation= 'softmax'))

# Compilando Rede Neural
classifier2.compile(optimizer= 'adam', loss= 'categorical_crossentropy'
                    , metrics= ['accuracy'])

# Treinando e validando modelo
classifier2.fit(x= X_train_codificados, y= y_train_dummy, batch_size= 256
                , epochs= 500, validation_data= (X_test_codificados, y_test_dummy))

loss, accuracy = classifier2.evaluate(X_test_codificados, y_test_dummy)
