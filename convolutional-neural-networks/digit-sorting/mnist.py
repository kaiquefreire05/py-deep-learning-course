#%% Importações

import matplotlib.pyplot as plt
from keras.datasets import mnist  # Importando os dados
from keras.models import Sequential  # usar Redes Neurais em modelos sequenciais
from keras.layers import Dense  # Usar camadas densas nas Redes Neurais
from keras.layers import Flatten  # conversão de dados em vetor
from keras.layers import Conv2D  # extração de características de imagens
from keras.layers import MaxPooling2D  # redução de dimensionalidade
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.utils import to_categorical

#%% Carregando base de dados

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#plt.imshow(x_train[2], cmap= 'gray')

"""
Reestrutura as imagens de treinamento e teste para o formato (28, 28, 1)
onde 28x28 é a dimensão da imagem e 1 é o número de canais (escala de cinza)
"""
previsores_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
previsores_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Converte os valores dos pixels para o tipo float32
previsores_train = previsores_train.astype('float32')
previsores_test = previsores_test.astype('float32')

# Normaliza os valores dos pixels para o intervalo [0, 1]
previsores_train /= 255
previsores_test /= 255

""" 
Converte os rótulos das imagens de treinamento e teste para o formato one-hot encoding
Isso é necessário para usar as classes como rótulos em um modelo de classificação
Por exemplo, se uma imagem contém o dígito 5, o rótulo correspondente será [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
Onde o valor 1 está na posição correspondente ao dígito 5
"""
classe_treinamento = to_categorical(y_train, 10)
classe_teste = to_categorical(y_test, 10)
"""
10 é o número de classes no conjunto de dados MNIST (dígitos de 0 a 9)
A função to_categorical converte os rótulos em um formato binário de 
categoria, necessário para o treinamento de modelos de classificação.
"""

#%% Estrutura da Rede Neural

# Inicialização do classificador como uma sequência de camadas
classificador = Sequential()

""" Primeira Camada de Convolução """
# Adicionando a camada de convolução 2D com 32 filtros de tamanho 3x3
# e função de ativação ReLU, com input_shape correspondente às dimensões das imagens (28x28 pixels, 1 canal)
classificador.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))

# Normalização de batch
classificador.add(BatchNormalization())

# Adicionando a camada de MaxPooling para redução de dimensionalidade
classificador.add(MaxPooling2D(pool_size=(2, 2)))

""" Segunda Camada de Convolução"""
# Adicionando outra camada de convolução
classificador.add(Conv2D(32, (3, 3), activation= 'relu'))

# Adicionando outro BatchNormalization
classificador.add(BatchNormalization())

# Adicionando outro Pooling
classificador.add(MaxPooling2D(pool_size= (2, 2)))

# Adicionando a camada de Flatten para converter os mapas de características em um vetor unidimensional
classificador.add(Flatten())

""" Redes Neurais Densas """

# Adicionando a camada densa (totalmente conectada) com 128 neurônios e função de ativação ReLU
classificador.add(Dense(units=128, activation='relu'))

# Camada de Dropout
classificador.add(Dropout(0.2))

# Segunda Camada Oculta
classificador.add(Dense(units= 128, activation= 'relu'))

# Camada de Dropout
classificador.add(Dropout(0.2))

# Adicionando a camada de saída com 10 neurônios (um para cada classe) e função de ativação softmax
classificador.add(Dense(units=10, activation='softmax'))

# Compilando o classificador, configurando o otimizador 'adam' e a função de perda 'categorical_crossentropy'
classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])

# Treinando o classificador com os dados de treinamento, utilizando um batch_size de 128 e 5 épocas
# e validando com os dados de teste
classificador.fit(previsores_train, classe_treinamento, batch_size=128, epochs=5, validation_data=(previsores_test, classe_teste))

# Avaliando o desempenho do classificador com os dados de teste
resultado = classificador.evaluate(previsores_test, classe_teste)