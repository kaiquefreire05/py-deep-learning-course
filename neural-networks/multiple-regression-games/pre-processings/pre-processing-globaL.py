#%% Importações

import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

#%% Carregando dataframe

df = pd.read_csv('C:/Users/kaiqu/OneDrive/Documentos/py-deep-learning/neural-networks/multiple-regression-games/dataset/games.csv') # carregando base de dados

df.drop('NA_Sales', axis= 1, inplace= True) # colunas desnecessárias
df.drop('EU_Sales', axis= 1, inplace= True)
df.drop('JP_Sales', axis= 1, inplace= True)
df.drop('Other_Sales', axis= 1, inplace= True)
df.drop('Developer', axis= 1, inplace= True)

df.dropna(axis= 0, inplace= True) # removendo as linhas nulas do dataframe

nome_jogos = df['Name'] # armazenando o nome dos jogos
df.drop('Name', axis= 1, inplace= True) # removendo o nome dos jogos do dataframe original

#%% Removendo valores de vendas que são menores ou igual a 1

small_sales = df.loc[df['Global_Sales'] <= 0.5]
df = df.loc[df['Global_Sales'] > 2.5]

#%% Fazendo transformação das colunas

encoder = LabelEncoder()

# 0, 2, 3, 9

colunas_encodar = ['Platform', 'Genre', 'Publisher', 'Rating']

for i in colunas_encodar:
    df[i] = encoder.fit_transform(df[i])
    
df_dummies = pd.get_dummies(data= df, columns= ['Platform', 'Genre', 'Publisher', 'Rating'])

#%% Dividindo previsoes e classe

x = df_dummies.drop('Global_Sales', axis= 1).values
y = df_dummies['Global_Sales'].values

#%% Salvando variáveis

with open('games_global.pkl', 'wb') as f:
    pickle.dump([x, y], f)
    