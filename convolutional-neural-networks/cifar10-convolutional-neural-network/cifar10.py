#%% Importações

from keras.datasets import cifar10  # variáveis de treino e teste
from keras.utils import to_categorical  # transformar array padrão OneHot
from keras.layers import Dense  # camada de rede neural densa
from keras.layers import Dropout  # camada de dropout para evitar overfitting
from keras.layers import Conv2D  # camada de convolução para extrair dados das imagens
from keras.layers import MaxPooling2D  # redução de dimensionalidade da imagem
from keras.layers import Flatten   # transformar em array
from keras.layers import BatchNormalization  # evitar overfitting em imagens
from keras.models import Sequential  # modelo sequencial de rede neural

from keras.preprocessing.image import ImageDataGenerator  # gerar novas imagens

#%% Carregamento e pré-processamento do dataframe

# Carregando os dados
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

# 32 x 32 x 3

# Reestruturando as imagens e sua escala de cor

x_train.reshape(x_train.shape[0], 32, 32, 3) 
x_test.reshape(x_test.shape[0], 32, 32, 3)

# Mudando o tipo primitivo para float

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizando os pixels no intervalo [0, 1]

x_train /= 255
x_test /= 255

# Processando classe para o formato OneHot

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#%% Criando estrutura da Rede Neural

# Modelo de Rede Sequencial
classificador = Sequential()

# Primeira parte
# Camada de extração de características
classificador.add(Conv2D(filters= 32, kernel_size= (3, 3), activation= 'relu', input_shape= (32, 32, 3)))

# Camada de normalização (evitar overfitting)
classificador.add(BatchNormalization())

# Camada para reduzir a dimensionalidade das imagens
classificador.add(MaxPooling2D(pool_size= (2, 2)))

# Camada de Dropout
classificador.add(Dropout(rate= 0.25))

# Segunda parte
# Camada de extração de características
classificador.add(Conv2D(filters= 32, kernel_size= (3, 3), activation= 'relu'))

# Camada de normalização
classificador.add(BatchNormalization())

# Camada de extração de características
classificador.add(Conv2D(filters= 32, kernel_size= (3, 3), activation= 'relu'))

# Camada de normalização
classificador.add(BatchNormalization())

# Camada de redução de dimensionalidade
classificador.add(MaxPooling2D(pool_size= (2, 2)))

# Transformando dados em arrays
classificador.add(Flatten())

# Terceira parte

classificador.add(Dense(units= 128, activation= 'relu'))
classificador.add(Dropout(rate= 0.25))
classificador.add(Dense(units= 128, activation= 'relu'))
classificador.add(Dropout(rate= 0.25))
classificador.add(Dense(units= 10, activation= 'softmax'))

# Compilando e treinando
classificador.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics= ['accuracy'])

classificador.fit(x= x_train, y= y_train, batch_size= 128, epochs= 15, validation_data=(x_test, y_test))


