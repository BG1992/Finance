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
tick = 5
df['Delay'] = df['Full_Date'] - df['Full_Date'].shift(-tick)

df = df.iloc[::tick]

df.sort_values(by='Full_Date', inplace=True, ascending = True)
df.reset_index(inplace=True, drop=True)
print(df.shape)
df = df[df['Delay'] == dt.timedelta(minutes = -tick)][['Full_Date', 'Close']]
print(df.shape)

#df = df.iloc[:1000]

df['SMA'] = df['Close'].rolling(window=30).mean()
df['SMA_Diff'] = df['SMA'].shift(-1) - df['SMA']
df['SMA_Diff2'] = df['SMA_Diff'].shift(-1) - df['SMA_Diff']

pos_ct, ct = 0, 0
bought = False
capital = 1000
levar = 100
for i in range(30+10+2, df.shape[0]-100):
    d1 = df.iloc[i-11:i]['SMA_Diff'].tolist()
    d2 = df.iloc[i-11:i]['SMA_Diff2'].tolist()
    if not bought:
        if len(list(filter(lambda x: x < 0, d1))) <= 2:
            ct0 = len(list(filter(lambda x: abs(x) < pow(10,-7), d2)))
            ct_pos = len(list(filter(lambda x: x >= pow(10,-7), d2)))
            ct_neg = len(list(filter(lambda x: x <= -pow(10,-7), d2)))
            if ct_neg < 3 and ct_pos > 3:
                o = df.iloc[i]['Close']
                bought = True
    else:
        if df.iloc[i]['Close'] - o > 80*pow(10,-5):
            c = df.iloc[i]['Close']
            capital += (c/o-1)*levar*capital
            pos_ct += 1
            ct += 1
            print(i, capital, o, c, c / o)
            bought = False
        elif df.iloc[i]['Close'] - o < -40*pow(10,-5):
            c = df.iloc[i]['Close']
            capital += (c/o-1)*levar*capital
            ct += 1
            print(i, capital, o, c, c / o)
            bought = False
            # ct0 = len(list(filter(lambda x: abs(x) < pow(10,-7), d2)))
            # ct_pos = len(list(filter(lambda x: x >= pow(10,-7), d2)))
            # ct_neg = len(list(filter(lambda x: x <= -pow(10,-7), d2)))
            # if ct_neg >= 2 and ct_pos < 3:
            #     c = df.iloc[i]['Close']
            #     bought = False
            #     if c > o:
            #         pos_ct += 1
            #     capital *= (c/o)
            #     ct += 1
            #     print(i, capital, o, c, c/o)

print(capital)
print(pos_ct, ct, pos_ct/ct)