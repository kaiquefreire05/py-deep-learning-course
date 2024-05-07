#%% Importações

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense

#%% Carregando base de dados

(X_train, _), (X_test, _) = mnist.load_data()
# 28 * 28 = 784px

# Convertendo para o tipo float32 e normalizando os valores
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Transformando as dimensões do array
X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))

#%% Criação do AutoEncoder

# 784 entrada, 32 escondida, 784 saída
fator_compactacao = 784 / 32

# Criando modelo sequencial
autoencoder = Sequential()

# Adicionando camada de entrarda e oculta
autoencoder.add(Dense(units= 32, activation= 'relu', input_dim= 784))

# Camada de saída
autoencoder.add(Dense(units= 784, activation= 'sigmoid'))

# Sumário do AutoEncoder
autoencoder.summary()

# Compilando AutoEncoder
autoencoder.compile(optimizer= 'adam', metrics= ['accuracy']
                    , loss= 'binary_crossentropy')

# Treinando e validando AutoEncoder
autoencoder.fit(x= X_train, y= X_train, batch_size= 256, epochs= 100
                , validation_data= (X_test, X_test))

#%% Visualização das imagens

dimensao_original = Input(shape= (784, 32))
camada_encoder = autoencoder.layers[0]
encoder = Model(inputs= dimensao_original, outputs= camada_encoder(dimensao_original))
encoder.summary()

imagens_codificadas = encoder.predict(X_test)
imagens_decodificadas = autoencoder.predict(X_test)

numero_imagens = 10
imagens_teste = np.random.randint(X_test.shape[0], size= numero_imagens)

# Criando figura
plt.figure(figsize= (18, 18))
for i, indice_imagem in enumerate(imagens_teste):
    
    # imagem original
    eixo = plt.subplot(10,10,i + 1)
    plt.imshow(X_test[indice_imagem].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
    
    # imagem codificada
    eixo = plt.subplot(10,10,i + 1 + numero_imagens)
    plt.imshow(imagens_codificadas[indice_imagem].reshape(8,4))
    plt.xticks(())
    plt.yticks(())
    
     # imagem reconstruída
    eixo = plt.subplot(10,10,i + 1 + numero_imagens * 2)
    plt.imshow(imagens_decodificadas[indice_imagem].reshape(28,28))
    plt.xticks(())
    plt.yticks(())