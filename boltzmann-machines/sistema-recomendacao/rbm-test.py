from rbm import RBM
import numpy as np

# Instânciando o RBM, com 6 camdas visíveis e 3 ocultas
rbm = RBM(num_visible= 6, num_hidden= 3)

# Base de dados 
base = np.array([[0,1,1,1,0,1],
                 [1,1,0,1,1,1],
                 [0,1,0,1,0,1],
                 [0,1,1,1,0,1], 
                 [1,1,0,1,0,1],
                 [1,1,0,1,1,1]])

# Tags dos filmes
filmes = ['Freddy X Jason', 'O Ultimato Bourne', 'Star Trek'
          , 'Exterminador do Futuro', 'Norbit', 'Star Wars']

# Treinando com a base e com no máximo 5000 épocas 
rbm.train(data= base, max_epochs= 5000)

# Variável que contém os pesos do algoritmo
pesos = rbm.weights

# Registro que vai ser usado
leonardo = np.array([[0, 1, 0, 1, 0, 0]])

# Variável que recebe quais dos neurônios foram ativados
camada_escondida = rbm.run_visible(leonardo)

# Variável que contém os filmes recomendados
recomendacao = rbm.run_hidden(camada_escondida)
for i in range(len(leonardo[0])):
    if leonardo[0, i] == 0 and recomendacao[0, i] == 1:
        print(filmes[i])

