import pandas as pd
import os
import datetime as dt
from sklearn import linear_model
import numpy as np
from functools import reduce

folder_path = r'C:\Users\Marzena\PycharmProjects\DS\Algo'

dfs = []
for f in os.listdir(folder_path):
    dfs.append(pd.read_csv(os.path.join(folder_path, f), sep = ',',
                names = ['Date', 'Hour', 'Open', 'High', 'Low', 'Close', 'Volume']))

df = pd.concat(dfs, ignore_index=True)
dfs = None
df['Full_Date'] = pd.to_datetime(df['Date'] + df['Hour'], format='%Y.%m.%d%H:%M')
df['Delay'] = df['Full_Date'] - df['Full_Date'].shift(-1)

df = df.iloc[::1]

df.sort_values(by='Full_Date', inplace=True, ascending = True)
df.reset_index(inplace=True, drop=True)
print(df.shape)
df = df[df['Delay'] == dt.timedelta(minutes = -1)][['Full_Date', 'Close']]
print(df.shape)

#df = df.iloc[:1000]

df['SMA20'] = df['Close'].rolling(window=20).mean()
df['SMA60'] = df['Close'].rolling(window=60).mean()

df['SMA_Diff_curr'] = df['SMA60'] - df['SMA20']
df['SMA_Diff_prev'] = df['SMA60'].shift(-1) - df['SMA20'].shift(-1)

def populate_signal(row):
    if row['SMA_Diff_prev'] > 0 and row['SMA_Diff_curr'] < 0: return 'B'
    if row['SMA_Diff_prev'] < 0 and row['SMA_Diff_curr'] > 0: return 'S'

df['Signal'] = df.apply(lambda row: populate_signal(row), axis = 1)

capital = 1000
bought = False
pos_ct, ct = 0, 0
for i in range(60, df.shape[0]-100):
    if df.iloc[i]['Signal'] == 'B' and not bought:
        o, ind_b = df.iloc[i]['Close'], i
        bought = True
    elif df.iloc[i]['Signal'] == 'S' and bought:
        c = df.iloc[i]['Close']
        bought = False
        capital *= (c/o)
        ct += 1
        if c > o: pos_ct += 1
        print(ind_b, i, o, c, c/o - 1, capital)

print(capital)
print(pos_ct, ct, pos_ct/ct)