"""
    Desenvolver uma Rede Neural Convolucional para classificar o Homer e Bart.
    Criar camadas para extrair as características das imagens.
"""

#%% Importações

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator

#%% Estrutura da Rede Neural

classificador = Sequential([
    
    # Camada para extrair as características
    Conv2D(filters= 32, kernel_size= (3, 3), activation= 'relu', input_shape= (64, 64, 3)),
    
    # Camada para normalizar os Batchs (agilizar o treinamento e melhorar os resultados)
    BatchNormalization(),
    
    # Camada para reduzir a dimensionalidade
    MaxPooling2D(pool_size= (2, 2)),
    
    # Normalizar os Batchs 
    BatchNormalization(),
    
    # Extração de características
    Conv2D(filters= 32, kernel_size= (3, 3), activation= 'relu'),
    
    # Normalizar os batchs
    BatchNormalization(),
    
    # Reduzir dimensionalidade
    MaxPooling2D(pool_size= (2, 2)),
    
    # Transformar em formatado que o TensorFlow trabalha
    Flatten(),
    
    # Camada densa
    Dense(units= 128, activation= 'relu'),
    
    # Camada de dropout (evitar overfitting)
    Dropout(rate= 0.2),
    
    # Camada densa
    Dense(units= 128, activation= 'relu'),
    
    # Camada dropout (evitar overfitting)
    Dropout(rate= 0.2),

    # Camada densa
    Dense(units= 128, activation= 'relu'),
    
    # Camada dropout (evitar overfitting)
    Dropout(rate= 0.2),
    
    # Camada de saída
    Dense(units= 1, activation= 'sigmoid')
    
    ])

# Compilando Rede Neural
classificador.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

# Criando gerador e padronizador de imagens
generator_train = ImageDataGenerator(rescale = 1./255, rotation_range=7, 
                                         horizontal_flip = True, shear_range=0.2,
                                         height_shift_range=0.07, zoom_range=0.2)

generator_test = ImageDataGenerator(rescale= 1./255)

# Transformando e extraindo as imagens
base_train = generator_train.flow_from_directory('datasets/training_set', 
                                    target_size= (64, 64), batch_size= 10,
                                    class_mode= 'binary')

base_test = generator_test.flow_from_directory('datasets/test_set',
                                   target_size= (64, 64), batch_size= 10,
                                   class_mode= 'binary')

# Treianando Rede Neural 

classificador.fit_generator(base_train, steps_per_epoch= 196 / 10, epochs= 300,
                            validation_data= base_test, validation_steps= 73 / 10)

# val_accuracy: 0.8767
