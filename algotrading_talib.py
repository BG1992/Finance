import pandas as pd
import os
import datetime as dt
from sklearn import linear_model
import numpy as np
from functools import reduce
from random import randint

folder_path = r'C:\Users\Marzena\PycharmProjects\DS\Algo'

results = {}
spreads = {'EURUSD': 0.00009, 'GBPUSD': 0.00022, 'USDCHF': 0.00019, 'AUDUSD': 0.00013, 'EURGBP': 0.00021}
for pair in spreads:
    dfs = []
    results[pair] = []
    for f in os.listdir(os.path.join(folder_path, pair)):
        dfs.append(pd.read_csv(os.path.join(folder_path, pair, f), sep = ',',
                    names = ['Date', 'Hour', 'Open', 'High', 'Low', 'Close', 'Volume']))

    df_main = pd.concat(dfs, ignore_index=True)
    dfs = None
    df_main['Full_Date'] = pd.to_datetime(df_main['Date'] + df_main['Hour'], format='%Y.%m.%d%H:%M')
    tick = 30
    df_main['Delay'] = df_main['Full_Date'] - df_main['Full_Date'].shift(-tick)

    for k in range(100):

        #a, b = 0, df_main.shape[0]
        a, b = randint(0, df_main.shape[0]-1), randint(0, df_main.shape[0]-1)
        while not (a < b and b-a < 200000 and b-a > 100000):
            a, b = randint(0, df_main.shape[0]-1), randint(0, df_main.shape[0]-1)
        df = df_main.iloc[a:b]
        df = df.iloc[::tick]

        df.sort_values(by='Full_Date', inplace=True, ascending = True)
        df.reset_index(inplace=True, drop=True)
        #print(df.shape, df.iloc[0]['Full_Date'], df.iloc[df.shape[0]-1]['Full_Date'])
        df = df[df['Delay'] == dt.timedelta(minutes = -tick)][['Full_Date', 'Close']]

        #df = df.iloc[:1000]

        df['SMA'] = df['Close'].rolling(window=30).mean()
        df['SMA_Diff'] = df['SMA'].shift(-1) - df['SMA']
        df['SMA_Diff2'] = df['SMA_Diff'].shift(-1) - df['SMA_Diff']

        pos_ct, ct = 0, 0
        bought = False
        capital = 1000
        levar = 30
        for i in range(30+10+2, df.shape[0]-100):
            d1 = df.iloc[i-11:i]['SMA_Diff'].tolist()
            d2 = df.iloc[i-11:i]['SMA_Diff2'].tolist()
            if not bought:
                if len(list(filter(lambda x: x < 0, d1))) <= 2:
                    ct0 = len(list(filter(lambda x: abs(x) < pow(10,-7), d2)))
                    ct_pos = len(list(filter(lambda x: x >= pow(10,-7), d2)))
                    ct_neg = len(list(filter(lambda x: x <= -pow(10,-7), d2)))
                    if ct_neg < 4 and ct_pos > 3:
                        o = df.iloc[i]['Close']+spreads[pair]/2
                        bought = True
            else:
                if df.iloc[i]['Close']-spreads[pair]/2 - o > 120*pow(10,-5):
                    c = df.iloc[i]['Close']-spreads[pair]/2
                    capital += (c/o-1)*levar*capital
                    pos_ct += 1
                    ct += 1
                    print(i, capital, o, c, c / o)
                    bought = False
                elif df.iloc[i]['Close']-spreads[pair]/2 - o < -60*pow(10,-5):
                    c = df.iloc[i]['Close']-spreads[pair]/2
                    capital += (c/o-1)*levar*capital
                    ct += 1
                    print(i, capital, o, c, c / o)
                    bought = False
            if capital < 0: break

        # print(pair, k, capital, pos_ct, ct,
        #       df.shape, df.iloc[0]['Full_Date'], df.iloc[df.shape[0]-1]['Full_Date'])
        results[pair].append(capital)
    print(pair, min(results[pair]), max(results[pair]), np.mean(results[pair]), np.std(results[pair]))
