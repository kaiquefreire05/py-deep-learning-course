#%% Importações 

import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from keras.models import Sequential
from keras.layers import Dense

#%% Carregando dataframe e dropando colunas não usadas

df = pd.read_csv('dataset/autos.csv', encoding= 'ISO-8859-1')

df.drop('dateCrawled', axis= 1, inplace= True)
df.drop('dateCreated', axis= 1, inplace= True)
df.drop('nrOfPictures', axis= 1, inplace= True)
df.drop('postalCode', axis= 1, inplace= True)
df.drop('lastSeen', axis= 1, inplace= True)

df['name'].value_counts()
df.drop('name', axis= 1, inplace= True)

df['seller'].value_counts()
df.drop('seller', axis= 1, inplace= True)

df['offerType'].value_counts()
df.drop('offerType', axis= 1, inplace= True)

# Pré-processamento - Tratando valores inconsistentes

i1 = df.loc[df.price <= 10] # preço de carros por preços muito baixos
df = df[df.price > 10] # retirando os valores que são menores que 10

i2 = df.loc[df.price > 350000] 
df = df[df.price < 350000] # retirando os valores que são maiores que 350k

#%% Pré-processamento - Valores faltantes

df.loc[pd.isnull(df['vehicleType'])]
df['vehicleType'].value_counts() # limousine

df.loc[pd.isnull(df['gearbox'])]
df['gearbox'].value_counts() # manuell

df.loc[pd.isnull(df['model'])]
df['model'].value_counts() # golf

df.loc[pd.isnull(df['fuelType'])] # benzin
df['fuelType'].value_counts()

df.loc[pd.isnull(df['notRepairedDamage'])] # nein
df['notRepairedDamage'].value_counts()

valores = {'vehicleType': 'limousine', 'gearbox': 'manuell', 'model': 'golf', 'fuelType': 'benzin', 'notRepairedDamage': 'nein'}
df = df.fillna(value= valores)

#%% Pré-processamento - LabelEncoder

x = df.drop('price', axis= 1).values
y = df['price'].values

encoder_x = LabelEncoder() # instânciando objeto LabelEncoder
indices = [0, 1, 3, 5, 8, 9, 10] # indices que vão ser transformados
x_encoded = x.copy() # copiando o array para outra variável
 
for i in indices: # iterando sobre a lista de indices e tranformando o array
    x_encoded[:, i] = encoder_x.fit_transform(x[:, i])
    
#%% Pré-processamento - OneHotEncoder

onehot = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],remainder='passthrough')
x_final = onehot.fit_transform(x_encoded).toarray()

#%% Estrutura da Rede Neural

regressor = Sequential() # Instânciando Rede Neural sequencial

regressor.add(Dense(units= 158, activation= 'relu', input_dim= 316)) # camada de entrada da Rede Neural
regressor.add(Dense(units= 158, activation= 'relu')) # camada oculta da Rede Neural
regressor.add(Dense(units= 1, activation= 'linear')) # camada de saída
regressor.compile(loss= 'mean_absolute_error', optimizer= 'adam', metrics= ['mean_absolute_error']) # compilando Rede Neural
regressor.fit(x= x_final, y= y, epochs= 2000, batch_size= 300) # treinando Rede Neural

#%% Previsões

pred = regressor.predict(x_final)

#%% Salvando variáveis pré-processadas

with open('autos.pkl', 'wb') as f:
    pickle.dump([x_final, y], f)
