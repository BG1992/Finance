import pandas as pd
import os
import talib
from collections import deque
from random import randint
import numpy as np

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
    tick = 60
    df_main['Delay'] = df_main['Full_Date'] - df_main['Full_Date'].shift(-tick)

    for k in range(1):

        #a, b = 0, df_main.shape[0]
        a, b = randint(0, df_main.shape[0]-1), randint(0, df_main.shape[0]-1)
        while not (a < b and b-a < 800000 and b-a > 600000):
            a, b = randint(0, df_main.shape[0]-1), randint(0, df_main.shape[0]-1)
        print(pair, df_main.iloc[a]['Full_Date'], df_main.iloc[b]['Full_Date'])
        df = df_main.iloc[a:b]
        df = df.iloc[::tick]

        df.reset_index(inplace=True, drop=True)
        rsi = talib.RSI(df['Close'].values)

        df['RSI'] = rsi

        long_thres = 25
        short_thres = 75

        capital = 1000
        levar = 50
        position = []
        flag = None
        flag_rsi_long_open = deque(maxlen=3)
        flag_rsi_long_close = deque(maxlen=5)
        flag_rsi_short_open = deque(maxlen=3)
        flag_rsi_short_close = deque(maxlen=5)

        for i in range(df.shape[0]):
            if flag is None:
                flag_rsi_long_open.append(df.iloc[i]['RSI'])
                flag_rsi_short_open.append(df.iloc[i]['RSI'])
                try:
                    if flag_rsi_long_open[1] < long_thres and flag_rsi_long_open[0] > flag_rsi_long_open[1] \
                            and flag_rsi_long_open[2] > flag_rsi_long_open[1]:
                        flag = 'Long'
                        position = [df.iloc[i]['Full_Date'], 'Long', df.iloc[i]['Close'], df.iloc[i]['RSI']]
                        print(position)
                except:
                    pass
            elif flag == 'Long':
                flag_rsi_long_close.append(df.iloc[i]['RSI'])
                flag_rsi_short_open.append(df.iloc[i]['RSI'])
                tmp_close_flag = 0
                if len(flag_rsi_long_close) == 7:
                    for j in range(len(flag_rsi_long_close) - 1):
                        if flag_rsi_long_close[j] > flag_rsi_long_close[j + 1]:
                            tmp_close_flag += 1
                    if tmp_close_flag == 0:
                        capital += ((df.iloc[i]['Close']-spreads[pair]/2) / (position[2]+spreads[pair]/2)-1)*levar*capital
                        print([df.iloc[i]['Full_Date'], 'Long Closed', df.iloc[i]['Close'], df.iloc[i]['RSI'],
                               capital])
                        flag = None
                try:
                    if flag_rsi_short_open[1] > short_thres and flag_rsi_short_open[0] < flag_rsi_short_open[1] \
                            and flag_rsi_short_open[2] < flag_rsi_short_open[1]:
                        if flag == 'Long':
                            capital += ((df.iloc[i]['Close']-spreads[pair]/2) / (position[2]+spreads[pair]/2)-1)*levar*capital
                            print([df.iloc[i]['Full_Date'], 'Long Closed', df.iloc[i]['Close'], df.iloc[i]['RSI'],
                                   capital])
                        flag = 'Short'
                        position = [df.iloc[i]['Full_Date'], 'Short', df.iloc[i]['Close'], df.iloc[i]['RSI']]
                        print(position)
                except:
                    pass
            elif flag == 'Short':
                flag_rsi_short_close.append(df.iloc[i]['RSI'])
                flag_rsi_long_open.append(df.iloc[i]['RSI'])
                tmp_close_flag = 0
                if len(flag_rsi_short_close) == 7:
                    for j in range(len(flag_rsi_short_close) - 1):
                        if flag_rsi_short_close[j] < flag_rsi_short_close[j + 1]:
                            tmp_close_flag += 1
                    if tmp_close_flag == 0:
                        capital += ((position[2]-spreads[pair]/2) / (df.iloc[i]['Close']+spreads[pair]/2)-1)*levar*capital
                        print([df.iloc[i]['Full_Date'], 'Short Closed', df.iloc[i]['Close'], df.iloc[i]['RSI'],
                               capital])
                        flag = None
                try:
                    if flag_rsi_long_open[1] < long_thres and flag_rsi_long_open[0] > flag_rsi_long_open[1] \
                            and flag_rsi_long_open[2] > flag_rsi_long_open[1]:
                        if flag == 'Short':
                            capital += ((position[2]-spreads[pair]/2) / (df.iloc[i]['Close']+spreads[pair]/2)-1)*levar*capital
                            print([df.iloc[i]['Full_Date'], 'Short Closed', df.iloc[i]['Close'], df.iloc[i]['RSI'],
                                   capital])
                        flag = 'Long'
                        position = [df.iloc[i]['Full_Date'], 'Long', df.iloc[i]['Close'], df.iloc[i]['RSI']]
                        print(position)
                except:
                    pass
            if capital < 0: break
        results[pair].append(capital)
    print(pair, min(results[pair]), max(results[pair]), np.mean(results[pair]), np.std(results[pair]))