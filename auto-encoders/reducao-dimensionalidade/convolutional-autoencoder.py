#%% Importações

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Flatten
from keras.layers import Reshape

#%% Carregando e fazendo o pré-processamento dos dados

(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

#%% Criando AutoEncoder

autoencoder = Sequential()

# Encoder
# Camada de entrada e primeira camada oculta de convolução
autoencoder.add(Conv2D(filters= 16, kernel_size= (3, 3)
                       , activation= 'relu', input_shape= (28, 28, 1)))
autoencoder.add(MaxPooling2D(pool_size= (2, 2)))

# Segunda camada oculta e segunda camada de pooling
autoencoder.add(Conv2D(filters= 8, kernel_size= (3, 3)
                       , activation= 'relu', padding= 'same'))
autoencoder.add(MaxPooling2D(pool_size= (2, 2), padding= 'same'))

# Terceira camada de convolução
autoencoder.add(Conv2D(filters= 8, kernel_size= (3, 3)
                       , activation= 'relu', padding= 'same', strides= (2, 2)))

# Transforma em 128 dimensões
autoencoder.add(Flatten())

# Volta ao shape antes de usar o flatten
autoencoder.add(Reshape(target_shape= (4, 4, 8)))

# Decoder
autoencoder.add(Conv2D(filters= 8, kernel_size= (3, 3)
                       , activation= 'relu', padding= 'same'))
autoencoder.add(UpSampling2D(size= (2, 2)))

autoencoder.add(Conv2D(filters= 8, kernel_size= (3, 3)
                       , activation= 'relu', padding= 'same'))
autoencoder.add(UpSampling2D(size= (2, 2)))

# Primeiro e último não precisa do padding
autoencoder.add(Conv2D(filters= 16, kernel_size= (3, 3)
                       , activation= 'relu'))
autoencoder.add(UpSampling2D(size= (2, 2)))

autoencoder.add(Conv2D(filters= 1, kernel_size= (3, 3)
                       , activation= 'sigmoid', padding= 'same'))

# Sumário da Rede Neural
autoencoder.summary()

#%% Compilando e treinando AutoEncoder

autoencoder.compile(optimizer= 'adam', loss= 'binary_crossentropy'
                    , metrics= ['accuracy'])

autoencoder.fit(x= X_train, y= X_train, batch_size= 256, epochs= 50
                , validation_data= (X_test, X_test))

#%% Criando modelo para obter somente codificador
encoder = Model(inputs = autoencoder.input
                , outputs= autoencoder.get_layer('flatten_2').output)
encoder.summary()

imagens_cod = encoder.predict(X_test)
imagens_decod = autoencoder.predict(X_test)

#%% Visualização de imagens

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
    plt.imshow(imagens_cod[indice_imagem].reshape(16,8))
    plt.xticks(())
    plt.yticks(())
    
     # imagem reconstruída
    eixo = plt.subplot(10,10,i + 1 + numero_imagens * 2)
    plt.imshow(imagens_decod[indice_imagem].reshape(28,28))
    plt.xticks(())
    plt.yticks(())

