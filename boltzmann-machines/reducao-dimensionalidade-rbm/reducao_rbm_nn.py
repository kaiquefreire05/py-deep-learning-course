#%% Importações

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.pipeline import Pipeline

#%% Carregando dataset

base = datasets.load_digits()
previsores = np.asarray(base.data, dtype= 'float32')
classe = base.target

# Normalizando os dados

normalizer = MinMaxScaler(feature_range= (0, 1))
previsores = normalizer.fit_transform(X= previsores)

#%% Dividindo entre variáveis de treino e teste

X_train, X_test, y_train, y_test = train_test_split(previsores, classe, test_size= 0.2, random_state= 42)

#%% Criando RBM e treinando com uma Rede Neural Artificial 

rbm = BernoulliRBM()
rbm.n_components = 50
rbm.n_iter = 25

nn_classifier = MLPClassifier(hidden_layer_sizes= (100,)
                              , learning_rate= 'constant'
                              , batch_size= 128, max_iter= 200)

classificador_rbm = Pipeline(steps= [('rbm', rbm), ('rede_neural', nn_classifier)])
classificador_rbm.fit(X= X_train, y= y_train)

#%% Fazendo previsão com dimensionalidade reduzida

previsoes_rbm = classificador_rbm.predict(X_test)

# Obtendo accuracy
accuracy_rbm = accuracy_score(y_true= y_test, y_pred= previsoes_rbm)

#%% Fazendo previsão sem reduzir com RBM

nn_dense = MLPClassifier(hidden_layer_sizes= (100,)
                              , learning_rate= 'constant'
                              , batch_size= 128, max_iter= 200)

# Fazendo o treinamento 
nn_dense.fit(X= X_train, y= y_train)

#%% Fazendo previsões sem reduzir a dimensionalidade

previsoes_nn = nn_dense.predict(X_test)

# Obtendo o accuracy
accuracy_nn = accuracy_score(y_true= y_test, y_pred= previsoes_nn)

#%% Resultados

print(f'Os resultados da Rede Neural Densa com imagens reduzidas foi de: {accuracy_rbm}')
print(f'Os resultados da Rede Neural Densa sem reduzir foi de: {accuracy_nn}')
print(f'A diferença da reduzida e sem reduzir foi de: {accuracy_rbm - accuracy_nn}')
