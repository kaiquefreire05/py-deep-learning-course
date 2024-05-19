#%% Importações necessárias

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.layers import Input
from keras.models import Sequential
from keras.models import Model
from keras.datasets import cifar10

#%% Carregando e preparando as variáveis

(X_train, _), (X_test, _) = cifar10.load_data()

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))

#%% Função para criar AutoEncoder

def create_autoencoder():
    autoencoder = Sequential()
    
    # Encode
    autoencoder.add(Dense(units= 1536, activation= 'relu', input_dim= 3072))
    autoencoder.add(Dense(units= 768, activation= 'relu'))
    autoencoder.add(Dense(units= 384, activation= 'relu'))
    
    # Decode
    autoencoder.add(Dense(units= 768, activation= 'relu'))
    autoencoder.add(Dense(units= 1536, activation= 'relu'))
    autoencoder.add(Dense(units= 3072, activation= 'sigmoid'))
    autoencoder.compile(optimizer= 'adam', loss= 'binary_crossentropy'
                        , metrics= ['accuracy'])
    autoencoder.fit(x= X_train, y= X_train, batch_size= 512, epochs= 100
                    , validation_data= (X_test, X_test))
    
    return autoencoder

#%% Usando autoencoder

autoencoder = create_autoencoder()
autoencoder.summary()

#%% Criando um codificador somente para codificar

dimensao_original = Input(shape= (3072,))
cm1 = autoencoder.layers[0]
cm2 = autoencoder.layers[1]
cm3 = autoencoder.layers[2]

encoder = Model(dimensao_original, cm3(cm2(cm1(dimensao_original))))

imagens_cod = encoder.predict(X_test)
imagens_decod = autoencoder.predict(X_test)

#%% Plotando os gráficos 

num_imagens = 10
imagens_aleatorias = np.random.randint(X_test.shape[0], size= num_imagens)

plt.figure(figsize= (18, 18))
for i, indice_imagem in enumerate(imagens_aleatorias):
    
    # imagem original
    eixo = plt.subplot(10,10,i + 1)
    plt.imshow(X_test[indice_imagem].reshape(32,32,3))
    plt.xticks(())
    plt.yticks(())
    
    # imagem codificada
    eixo = plt.subplot(10,10,i + 1 + num_imagens)
    plt.imshow(imagens_cod[indice_imagem].reshape(24, 16))
    plt.xticks(())
    plt.yticks(())
    
     # imagem reconstruída
    eixo = plt.subplot(10,10,i + 1 + num_imagens * 2)
    plt.imshow(imagens_decod[indice_imagem].reshape(32,32,3))
    plt.xticks(())
    plt.yticks(())

