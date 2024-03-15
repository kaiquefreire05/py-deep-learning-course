#%% Importações

from keras.datasets import mnist  # base de dados
from keras.models import Sequential  # rede neural sequencial
from keras.layers import Dense  # camada densa (interligada em todos os neurônios)
from keras.layers import Conv2D  # extração de características
from keras.layers import MaxPooling2D  # redução de dimensionalidade
from keras.layers import Flatten  # transformar imagens em arrays
from keras.utils import to_categorical  # transformar padrão OneHot
from keras.preprocessing.image import ImageDataGenerator  # gerar novas imagens 

#%% Carregando base de dados

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#%% Mudando dimensão das imagens previsoras e transformando elas em cinza e 'float32'

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#%% Transformando a classe em array padrão OneHot com 10 classes

y_train = to_categorical(y= y_train, num_classes= 10)
y_test = to_categorical(y= y_test, num_classes= 10)

#%% Criando Rede Neural

# Criando Rede Neural Sequencial
classificador = Sequential()

# Adicionando camada de convolução para a extração de características
classificador.add(Conv2D(filters= 32, kernel_size= (3, 3), activation= 'relu',
                         input_shape= (28, 28, 1)))

# Criando camada de Pooling
classificador.add(MaxPooling2D(pool_size= (2, 2)))
 
# Criando camada de Flatten para transformar em arrays
classificador.add(Flatten())

# Adicionando camada densa
classificador.add(Dense(units= 128, activation= 'relu'))

# Adicionando camada densa de saída
classificador.add(Dense(units= 10, activation= 'softmax'))

# Compilando Rede Neural
classificador.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics= ['accuracy'])

#%% Argumentation (Geração de novas imagens)

gerador_train = ImageDataGenerator(rotation_range= 7, horizontal_flip= True, 
                                   shear_range= 0.2, height_shift_range= 0.07, 
                                   zoom_range= 0.2)

gerador_test = ImageDataGenerator()

train_base = gerador_train.flow(x= x_train, y= y_train, batch_size= 128)
test_base = gerador_test.flow(x= x_test, y= y_test, batch_size= 128)

classificador.fit_generator(train_base, steps_per_epoch= 60000 / 128, epochs= 5, 
                            validation_data= test_base, validation_steps= 10000 / 128)
