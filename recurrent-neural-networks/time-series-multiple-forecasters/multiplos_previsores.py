#%% Importações

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

#%% Carregando os datasets

base = pd.read_csv('datasets/petr4_treinamento.csv')
test = pd.read_csv('datasets/petr4_teste.csv')

# Removendo valores nulos
base = base.dropna()

# base_treinamento = base.drop('Date', axis= 1).values
train_base = base.iloc[:, 1:7].values

# Criando objeto normalizador e normalizando
normalizador = MinMaxScaler(feature_range= (0, 1))
train_base_normalized = normalizador.fit_transform(train_base)

normalizador_previsao = MinMaxScaler(feature_range= (0, 1))
normalizador_previsao.fit_transform(train_base[:, 0:1])

# Listas  que guardam os 90 previsores e o valor real
previsores = []
preco_real = []

for i in range(90, 1242):
    previsores.append(train_base_normalized[i-90:i, 0:6])
    preco_real.append(train_base_normalized[i, 0])

# Transformando em um NumpyArray
previsores, preco_real = np.array(previsores), np.array(preco_real)

#%% Estrutura da Rede Neural Recorrente

regressor = Sequential()
regressor.add(LSTM(units= 100, return_sequences= True, input_shape= (previsores.shape[1], 6)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units= 50, return_sequences= True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units= 50, return_sequences= True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units= 50, return_sequences= False))
regressor.add(Dropout(0.3))

regressor.add(Dense(units= 1, activation= 'sigmoid')) # 'linear', outra função
regressor.compile(optimizer= 'adam', loss= 'mean_squared_error', metrics= ['mean_absolute_error'])

#  Criando Callbacks

# EarlyStopping: interrompe o treinamento se a perda (loss) não melhorar após 10 épocas consecutivas
# min_delta especifica a mudança mínima considerada como melhoria
# verbose controla a quantidade de saída durante o treinamento
es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=1)

# ReduceLROnPlateau: reduz a taxa de aprendizado se a perda não melhorar após 5 épocas consecutivas
# factor especifica a taxa pela qual a taxa de aprendizado será reduzida (0.2 significa redução de 20%)
# verbose controla a quantidade de saída durante o treinamento
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)

# ModelCheckpoint: salva os pesos do modelo após cada época se houver uma melhoria na perda
# filepath especifica o local onde os pesos serão salvos
# monitora 'loss' para determinar se houve melhoria
# save_best_only=True garante que apenas o melhor modelo (com base na perda) seja salvo
mcp = ModelCheckpoint(filepath='pesos.h5', monitor='loss', save_best_only=True)

# Treinando
regressor.fit(previsores, preco_real, epochs= 100, batch_size= 32, callbacks= [es, rlr, mcp])

#%% Preparando base de teste

# Criando um array somente com os preços reais do atributo 'Open'
real_price_test = test['Open'].values

# Concatenando as bases de dados
frames = [base, test]
base_completa = pd.concat(frames)
base_completa.drop('Date', axis= 1, inplace= True)

entradas = base_completa[len(base_completa) - len(test) - 90:].values
entradas = normalizador.transform(entradas)

x_test = []

for i in range(90, 112):
    x_test.append(entradas[i-90:i, 0:6])
    
# Transformando em um NumpyArray
x_test = np.array(x_test)



previsoes = regressor.predict(x_test)
previsoes = normalizador_previsao.inverse_transform(previsoes)

print(previsoes.mean())
print(real_price_test.mean())

# Plotando os dados
plt.figure(figsize=(10, 6))
plt.plot(real_price_test, color='red', label='Preço Real', linestyle='-')
plt.plot(previsoes, color='blue', label='Previsões', linestyle='--')

# Destacando pontos de divergência
for i in range(len(real_price_test)):
    if real_price_test[i] != previsoes[i]:
        plt.scatter(i, real_price_test[i], color='red', marker='x', s=50)
        plt.scatter(i, previsoes[i], color='blue', marker='x', s=50)

# Adicionando título e rótulos dos eixos
plt.title('Comparação entre Preço Real e Previsões', fontsize=14, fontweight='bold')
plt.ylabel('Valor Yahoo', fontsize=10, fontweight='bold')
plt.xlabel('Tempo', fontsize=10, fontweight='bold')

# Adicionando legenda e grade
plt.legend()
plt.grid(True, linestyle='--')

# Exibindo o gráfico
plt.show()