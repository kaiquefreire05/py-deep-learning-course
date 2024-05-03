#%% Importações

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline # Serve para executar mais de uma coisa por vez


#%% Importando dataset

# Carregando dataset dos dígitos
base = datasets.load_digits()

# Pegando os previsores e classe da base
previsores = np.asarray(base.data, dtype= 'float32')
classe = base.target

# Normalizando os valores
normalizer = MinMaxScaler(feature_range= (0, 1))
previsores = normalizer.fit_transform(previsores)

# Dividindo entre variáveis de treino e teste

X_train, X_test, y_train, y_test = train_test_split(previsores, classe, test_size= 0.2, random_state= 42)

#%% Criando o RBM

rbm = BernoulliRBM()
rbm.n_iter = 25
rbm.n_components = 50 # Para reduzir a dimensionalidade

# Criando Naive Bayes
naive_rbm = GaussianNB()

# Criando Pipeline
classificador_rbm = Pipeline(steps= [('rbm', rbm), ('naive', naive_rbm)])
classificador_rbm.fit(X_train, y_train)

#%% Plotando figura

plt.figure(figsize= (20, 20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1) # Mostrando as imagens do i + 1 = 1
    plt.imshow(comp.reshape((8, 8)), cmap= plt.cm.gray_r) # Reshape e mudando a cor
    plt.xticks(()) # Removendo eixo X
    plt.yticks(()) # Removendo eixo Y
plt.show() # Plotando a imagem

#%% Fazendo previsões

previsoes_rbm = classificador_rbm.predict(X_test)
accuracy_rbm = metrics.accuracy_score(y_true= y_test, y_pred= previsoes_rbm)
print(f'O accuracy score reduzindo a dimensionalidade com rbm foi de: {accuracy_rbm}')

#%% Testando sem RBM

# Criando variável com o algoritmo Naive Bayes
naive_simples = GaussianNB()

# Fazendo o treinamento
naive_simples.fit(X= X_train, y= y_train)

# Obtendo as previsões
previsoes_naive = naive_simples.predict(X_test)

# Obtendo o accuracy
accuracy_naive = metrics.accuracy_score(y_true= y_test, y_pred= previsoes_naive)
print(f'O accuracy score do algoritmo sem reduzir a dimensionalidade foi de: {accuracy_naive}')
print(f'A diferença entre o accuracy com RBM e sem RBM foi de: {accuracy_rbm - accuracy_naive}')