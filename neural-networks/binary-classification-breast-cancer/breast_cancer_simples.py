#%% Importações 


import keras # biblioteca de aprendizado de máquina de alto nível
from keras.models import Sequential # Sequential é uma forma de criar modelos de rede neural onde as camadas são empilhadas sequencialmente.
from keras.layers import Dense # Dense é usada para criar camadas densas, ou seja, camadas totalmente conectadas, em uma rede neural.

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#%% -- Carregando as bases de dados --


previsores = pd.read_csv('datasets/entradas_breast.csv')
classes = pd.read_csv('datasets/saidas_breast.csv')


#%% Divindindo entre base de treino e teste

x_train, x_test, y_train, y_test = train_test_split(previsores, classes, test_size=0.25)


#%% -- Criando a Neural Network (camada de entrada) -- 

"""
    - Aqui estamos criando um modelo sequencial, o que significa que as camadas serão adicionadas uma após a outra.
    - Esta linha adiciona uma camada densa (totalmente conectada) à rede neural.
    - A camada tem 16 neurônios (units) e usa a função de ativação 'relu' (unidade linear retificada).
    - 'relu' é uma função de ativação comum que retorna 0 para valores negativos e o próprio valor para valores positivos.
    - O 'kernel_initializer' é usado para inicializar os pesos da camada.
    - Neste caso, estamos usando 'random_uniform', o que significa que os pesos serão inicializados aleatoriamente de uma distribuição uniforme.
    - 'input_dim' especifica a dimensão de entrada desta camada, que é 30 neste caso.
    - Isso significa que espera-se que cada entrada tenha 30 recursos.
"""

classificador = Sequential()
classificador.add(Dense(units= 16, activation= 'relu',
                        kernel_initializer= 'random_uniform', input_dim= 30))

#%% -- Criando camada oculta na Neural Network --

classificador.add(Dense(units= 16, activation= 'relu',
                        kernel_initializer= 'random_uniform'))

#%% -- Criando camadas de saída da Neural Network -- 

"""
    - Esta linha adiciona outra camada densa à rede neural.
    - Esta será a camada de saída da rede.
    - A camada tem apenas 1 neurônio (units), o que é comum em problemas de classificação binária.
    - A função de ativação 'sigmoid' é usada nesta camada.
    - 'sigmoid' é comumente usado na camada de saída para problemas de classificação
    binária, pois comprime os valores entre 0 e 1, representando probabilidades.
"""
classificador.add(Dense(units= 1, activation= 'sigmoid'))

#%% -- Parâmetros de otimizador --
"""
    - learning_rate: A taxa de aprendizado do otimizador, definida como 0.001.
    
    - weight_decay: Um fator de decaimento de peso, que é uma técnica para 
    regularizar os pesos da rede, reduzindo-os durante o treinamento. 
    Aqui, é definido como 0.01.
    
    - clipvalue: Um valor limite que limita os gradientes durante o treinamento.
    Se o valor absoluto de um gradiente for maior que esse limite, ele será 
    reduzido para o valor do limite. Aqui, é definido como 0.5.
"""
otimizador = keras.optimizers.Adam(learning_rate= 0.001, weight_decay= 0.01, clipvalue= 0.5)

#%% -- Compilando a Neural Network --

"""
Na linha 67: 
    Essa linha compila o modelo de rede neural. O otimizador é definido como 
    Adam ('adam'), a função de perda é definida como entropia cruzada binária 
    ('binary_crossentropy'), e a métrica para avaliação é a precisão binária 
    ('binary_accuracy').
    
Na linha 68:
    Treina o modelo. Os dados de treinamento (x_train e y_train) são fornecidos.
    O treinamento é realizado em lotes de tamanho 10 (batch_size=10) por 
    100 épocas (epochs=100). Durante o treinamento, os pesos do modelo são 
    ajustados para minimizar a função de perda usando o otimizador Adam. 
    A precisão binária é calculada como uma métrica durante o treinamento.
    Passando variável otimizador intânciada anteriormente 
"""

classificador.compile(optimizer= otimizador, loss= 'binary_crossentropy', metrics= ['binary_accuracy']) # Compilando a rede neural
classificador.fit(x_train, y_train, batch_size= 10, epochs= 100) # fazendo o treinamento de 100 epochs

#%% -- Obtendo o valor dos pesos --
pesos0 = classificador.layers[0].get_weights()
print(pesos0)
print(len(pesos0))
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()
#%% -- Fazendo previsões --

predicts = classificador.predict(x_test) # Prevendo os valores da base X de teste
predicts = (predicts > 0.5) # Transformando os resultados em binário
accuracy_model = accuracy_score(y_test, predicts) # Criando variável de accuracy
confusion_model = confusion_matrix(y_test, predicts) # Criando variável de matriz de confusão

print(f'A taxa de acerto do algoritmo foi de: {accuracy_model}')
print(confusion_model)

resultado = classificador.evaluate(x_test, y_test)

"""
    - Sem camadas ocultas o resultado foi muito melhor superando 90%
    - Não necessariamente tendo mais camadas a rede neural vai ser superior
"""