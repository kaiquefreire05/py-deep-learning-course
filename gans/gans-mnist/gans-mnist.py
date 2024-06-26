#%% Importações

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import InputLayer, Dense, Flatten, Reshape
from keras.regularizers import L1L2
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
# pip install --upgrade keras 

#%% Carregamento dos dados

(previsores_treinamento, _), (_, _) = mnist.load_data()

# Normalizando os dados
previsores_treinamento = previsores_treinamento.astype('float32') / 255


#%% Criando gerador e discriminador
# Gerador
gerador = Sequential()
gerador.add(Dense(units = 500, input_dim = 100, activation = 'relu', 
                  kernel_regularizer = L1L2(1e-5, 1e-5)))
gerador.add(Dense(units = 500, activation = 'relu', 
                  kernel_regularizer = L1L2(1e-5, 1e-5)))
gerador.add(Dense(units = 784, activation = 'sigmoid', kernel_regularizer = L1L2(1e-5, 1e-5)))
gerador.add(Reshape((28,28)))

# Discriminador
discriminador = Sequential()
discriminador.add(InputLayer(input_shape=(28,28)))
discriminador.add(Flatten())
discriminador.add(Dense(units = 500, activation = 'relu', kernel_regularizer = L1L2(1e-5, 1e-5)))
discriminador.add(Dense(units = 500, activation = 'relu', kernel_regularizer = L1L2(1e-5, 1e-5)))
discriminador.add(Dense(units = 1, activation = 'sigmoid', kernel_regularizer = L1L2(1e-5, 1e-5)))

gan = simple_gan(gerador, discriminador, normal_latent_sampling((100,)))
modelo = AdversarialModel(base_model = gan,
                          player_params = [gerador.trainable_weights, 
                                           discriminador.trainable_weights])
modelo.adversarial_compile(adversarial_optimizer = AdversarialOptimizerSimultaneous(),
                           player_optimizers = ['adam', 'adam'],
                           loss = 'binary_crossentropy')
modelo.fit(x = previsores_treinamento, y = gan_targets(60000), epochs = 100, batch_size = 256)

amostras = np.random.normal(size = (20,100))
previsao = gerador.predict(amostras)
for i in range(previsao.shape[0]):
    plt.imshow(previsao[i, :], cmap='gray')
    plt.show()