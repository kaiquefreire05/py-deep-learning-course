#%% Importações

from rbm import RBM
import numpy as np

#%% Criação RBM

rbm = RBM(num_visible= 6, num_hidden= 2)

# Usuários
base = np.array([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0 ,1]])

filmes = ['A bruxa', 'Invocação do Mal', 'O chamado', 'Se beber não case'
          , 'Gente grande', 'American pie']

rbm.train(base, max_epochs= 5000)
pesos = rbm.weights

#%% Criando novo registro

usuario = np.array([[1, 1, 0, 1, 0, 0]])
usuario2 = np.array([[0, 0, 0, 1, 1, 0]])

res = rbm.run_visible(usuario)
re2 = rbm.run_visible(usuario2)

camada_escondida = np.array([[0, 1]])
recomendacao = rbm.run_hidden(camada_escondida)

for i in range(len(usuario[0])):
    # print(usuario[0, i])
    if usuario[0, i] == 0 and recomendacao[0, i] == 1:
        print(filmes[i])