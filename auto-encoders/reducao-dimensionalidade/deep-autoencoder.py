#%% Importações

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

#%% Carregando e preparando dados

# Carregando variáveis
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizando as variáveis entre 0 e 1
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Categorizando as variáveis no padrão OneHot

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Mudando o formato das variáveis
X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))

#%% Criando autoencoder profundo

autoencoder = Sequential()

# Encode
autoencoder.add(Dense(units= 128, activation= 'relu', input_dim= 784))
autoencoder.add(Dense(units= 64, activation= 'relu'))
autoencoder.add(Dense(units= 32, activation= 'relu'))

# Decode
autoencoder.add(Dense(units= 64, activation= 'relu'))
autoencoder.add(Dense(units= 128, activation= 'relu'))
autoencoder.add(Dense(units= 784, activation= 'sigmoid'))


# Camada de saída
autoencoder.add(Dense(units= 784, activation= 'sigmoid'))

# Compilando autoencoder
autoencoder.compile(optimizer= 'adam', loss= 'binary_crossentropy'
                    , metrics= ['accuracy'])

# Sumário da Rede Neural
autoencoder.summary()

# Treinando e validando 
autoencoder.fit(x= X_train, y= X_train, batch_size= 256, epochs= 100
                , validation_data= (X_test, X_test))

#%% Criando modelo para obter somente codificador

dimensao_original = Input(shape=(784,))
camada_encoder1 = autoencoder.layers[0]
camada_encoder2 = autoencoder.layers[1]
camada_encoder3 = autoencoder.layers[2]

encoder = Model(dimensao_original
                , camada_encoder3(camada_encoder2(camada_encoder1(dimensao_original))))

encoder.summary()

# Somente codifica
imagens_cod = encoder.predict(X_test)

# Codifica e volta ao normal
imagens_decod = autoencoder.predict(X_test) 


#%% Visualização das imagens

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
    plt.imshow(imagens_cod[indice_imagem].reshape(8,4))
    plt.xticks(())
    plt.yticks(())
    
     # imagem reconstruída
    eixo = plt.subplot(10,10,i + 1 + numero_imagens * 2)
    plt.imshow(imagens_decod[indice_imagem].reshape(28,28))
    plt.xticks(())
    plt.yticks(())