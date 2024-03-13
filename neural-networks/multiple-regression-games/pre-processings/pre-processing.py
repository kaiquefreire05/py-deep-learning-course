#%% Importações

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

#%% Carregando dataframe e excluindo colunas indesejadas

df = pd.read_csv('dataset/games.csv')
df.drop('Other_Sales', axis= 1, inplace= True)
df.drop('Global_Sales', axis= 1, inplace= True)
df.drop('Developer', axis= 1, inplace= True)

# Tratamento de valores faltantes

df.dropna(axis= 0, inplace= True)

#%% Retirando valores que tem sales menores que 1

negative_na_sales = df.loc[df['NA_Sales'] < 1]
df = df.loc[df['NA_Sales'] > 1]
df = df[df['EU_Sales'] > 1]

#%% Apagando coluna dos nomes

nome_jogos = df['Name']
df.drop('Name', axis= 1, inplace= True)

#%% Dividindo classes e previsoes

x = df.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10, 11]].values
y_na = df.iloc[:, 4].values
y_eu = df.iloc[:, 5].values
y_jp = df.iloc[:, 6].values

#%% Aplicando OneHotEncoder e LabelEncoder nos previsores

encoder = LabelEncoder()

x[:, 0] = encoder.fit_transform(x[:, 0])
x[:, 2] = encoder.fit_transform(x[:, 2])
x[:, 3] = encoder.fit_transform(x[:, 3])
x[:, 8] = encoder.fit_transform(x[:, 8])

onehot = ColumnTransformer(transformers= [('OneHot', OneHotEncoder(), [0, 2, 3, 8])], remainder= 'passthrough')

x = onehot.fit_transform(x).toarray()

#%% Salvando variáveis

with open('variaveis_games', 'wb') as f:
    pickle.dump([x, y_na, y_eu, y_jp], f)
