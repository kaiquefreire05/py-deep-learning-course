""" Fazer a classificação do Homer e Bart sem usar Redes Neurais Convolucionais """

#%% Importações

import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#%% Carregamento e pré-processamento do dataset

df = pd.read_csv('datasets/personagens.csv')

# Divindo previsores e classe

x = df.drop('classe', axis= 1).values
y = df['classe'].values

# Fazendo o encoder das classes
""" 
Bart = 0 
Homer = 1
"""
encoder_classe = LabelEncoder()
y = encoder_classe.fit_transform(y)

# Variáveis de teste e treinamento

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state= 42)
print(f'Tamanho das variáveis de treino: {x_train.shape}, {y_train.shape}')
print(f'Tamanho das variáveis de teste: {x_test.shape}, {y_test.shape}')

#%% Criando a estrutura da Rede Neural

classificador = Sequential([
    Dense(units= 4, activation= 'relu', input_dim= 6),
    Dropout(rate= 0.1),
    Dense(units= 4, activation= 'relu'),
    Dropout(rate= 0.1),
    Dense(units= 1, activation= 'sigmoid')
    ])

# Compilando Rede Neural 
classificador.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['binary_accuracy'])

# Fazendo o treinamento 
classificador.fit(x= x_train, y= y_train, batch_size= 8, epochs= 2000)

# Obtendo resultados com as variáveis de teste
resultado = classificador.evaluate(x= x_test, y= y_test)

"""
    Perda: 0.6718974709510803
    Acurácia: 0.9324324131011963
"""