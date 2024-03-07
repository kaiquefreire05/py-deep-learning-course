import numpy as np


# Função de ativação do neurônio
def step_function(soma):
    if soma >= 1:
        return 1
    return 0


# Função Sigmoide
def sigmoid_function(soma):
    return 1 / (1 + np.exp(-soma))


# Função Tangente Hiperbólica
def tahn_function(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))


# Função Relu 
def relu_function(soma):
    if soma >= 0:
        return soma
    return 0
    

# Função Linear
def linear_function(soma):
    return soma
    

# Função SoftMax
def softmax_function(x):
    ex = np.exp(x)

    return ex / ex.sum()
    
# Testes

teste = step_function(0.8)
teste = sigmoid_function(0.358)
teste = tahn_function(-0.358)
teste = relu_function(-0.358)
teste = linear_function(0.358)
