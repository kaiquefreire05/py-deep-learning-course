#%% Importações

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

#%% Estrutura da Rede Neural

# Criando modelo sequencial de Rede Neural
classificador = Sequential()

# Primeira Parte - Camada de extração de características
classificador.add(Conv2D(filters= 32, kernel_size= (3, 3), activation= 'relu', input_shape= (64, 64, 3)))

# Camada para normalizar os batchs e acelerar treinamento e resultado
classificador.add(BatchNormalization())

# Camada para reduzir a dimensionalidade
classificador.add(MaxPooling2D(pool_size= (2, 2)))

# Segunda Parte - Camada de extração de características
classificador.add(Conv2D(filters= 32, kernel_size= (3, 3), activation= 'relu'))

# Camada de normalização de batchs para acelerar treinamento e melhorar o resultado
classificador.add(BatchNormalization())

# Camada de redução de dimensionalidade
classificador.add(MaxPooling2D(pool_size= (2, 2)))

# Camada de Flatten (transformar matriz em vetor, e depois mandar para a Rede Neural densa)
classificador.add(Flatten())

# Terceira Parte - Criando camada densa
classificador.add(Dense(units= 128, activation= 'relu'))

# Inserindo camada de Dropout para evitar overfitting
classificador.add(Dropout(rate= 0.2))

# Adicionando mais uma camada densa
classificador.add(Dense(units= 128, activation= 'relu'))

# Camada de dropout (evitar overfitting)
classificador.add(Dropout(rate= 0.2))

# Camada de saída
"""
    como tem apenas 2 resultados basta colocar apenas units= 1
    temos apenas duas saídas (gato ou cachorro), então usar a ativação sigmoid (0, 1)
"""
classificador.add(Dense(units= 1, activation= 'sigmoid'))

# Compilando Rede Neural
classificador.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

#%% Gerando novas imagens e mudando sua escala [0, 1]

gerador_train = ImageDataGenerator(rescale= 1./255, rotation_range=  7,
                                   horizontal_flip= True, shear_range= 0.2,
                                   height_shift_range= 0.07, zoom_range= 0.2)

gerador_test = ImageDataGenerator(rescale= 1./255)

base_train = gerador_train.flow_from_directory('dataset/training_set',
                                            target_size= (64, 64),
                                            batch_size= 32, class_mode= 'binary')

base_test = gerador_test.flow_from_directory('dataset/test_set', target_size= (64, 64),
                                             batch_size= 32, class_mode= 'binary')

#%% Fazendo treinamento

classificador.fit_generator(base_train, steps_per_epoch= 4000 / 32,
                            epochs= 50, validation_data= base_test,
                            validation_steps= 1000 / 32)

#%% Previsão com somente uma imagem

imagem_teste = image.load_img('dataset/training_set/gato/cat.907.jpg', target_size= (64, 64)) # carregando a imagem
imagem_teste = image.img_to_array(imagem_teste) # transformando em um array
imagem_teste /= 255 # normalizando a imagem
imagem_teste = np.expand_dims(imagem_teste, axis= 0) # expandindo o array para o formato que o tensorflow usa as imagens

previsao = classificador.predict(imagem_teste)

imagem_teste_dog = image.load_img('dataset/test_set/cachorro/dog.3904.jpg', target_size= (64, 64))
imagem_teste_dog = image.img_to_array(imagem_teste_dog)
imagem_teste_dog /= 255
imagem_teste_dog = np.expand_dims(imagem_teste_dog, axis= 0)

previsao2 = classificador.predict(imagem_teste_dog)