import pandas as pd
import talib
import os
import datetime as dt
import numpy as np
from collections import deque

pd.set_option('display.expand_frame_repr', False)
folder_path = r'C:\Users\Marzena\PycharmProjects\DS\Algo'
df = os.path.join(folder_path, 'USDJPY', 'DAT_MT_USDJPY_M1_2021.csv')
df = pd.read_csv(df, sep = ',', names = ['Date', 'Hour', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['Full_Date'] = pd.to_datetime(df['Date'] + df['Hour'], format='%Y.%m.%d%H:%M')
tick = 60
df['Delay'] = df['Full_Date'] - df['Full_Date'].shift(-tick)
df2 = df.iloc[::tick]
df2.reset_index(inplace = True, drop = True)
df2.head()

rsi = talib.RSI(df2['Close'].values)

df2['RSI'] = rsi

long_thres = 30
short_thres = 70

capital = 1000
position = []
flag = None
flag_rsi_long_open = deque(maxlen=3)
flag_rsi_long_close = deque(maxlen=7)
flag_rsi_short_open = deque(maxlen=3)
flag_rsi_short_close = deque(maxlen=7)

for i in range(df2.shape[0]):
    if flag is None:
        flag_rsi_long_open.append(df2.iloc[i]['RSI'])
        flag_rsi_short_open.append(df2.iloc[i]['RSI'])
        try:
            if flag_rsi_long_open[1] < long_thres and flag_rsi_long_open[0] > flag_rsi_long_open[1] \
                and flag_rsi_long_open[2] > flag_rsi_long_open[1]:
                flag = 'Long'
                position = [df2.iloc[i]['Full_Date'], 'Long', df2.iloc[i]['Close'], df2.iloc[i]['RSI']]
                print(position)
        except: pass
    elif flag == 'Long':
        flag_rsi_long_close.append(df2.iloc[i]['RSI'])
        flag_rsi_short_open.append(df2.iloc[i]['RSI'])
        tmp_close_flag = 0
        if len(flag_rsi_long_close) == 7:
            for j in range(len(flag_rsi_long_close)-1):
                if flag_rsi_long_close[j] > flag_rsi_long_close[j+1]:
                    tmp_close_flag += 1
            if tmp_close_flag == 0:
                capital *= (df2.iloc[i]['Close'] / position[2])
                print([df2.iloc[i]['Full_Date'], 'Long Closed', df2.iloc[i]['Close'], df2.iloc[i]['RSI'], capital])
                flag = None
        try:
            if flag_rsi_short_open[1] > short_thres and flag_rsi_short_open[0] < flag_rsi_short_open[1] \
                and flag_rsi_short_open[2] < flag_rsi_short_open[1]:
                if flag == 'Long':
                    capital *= (df2.iloc[i]['Close'] / position[2])
                    print([df2.iloc[i]['Full_Date'], 'Long Closed', df2.iloc[i]['Close'], df2.iloc[i]['RSI'], capital])
                flag = 'Short'
                position = [df2.iloc[i]['Full_Date'], 'Short', df2.iloc[i]['Close'], df2.iloc[i]['RSI']]
                print(position)
        except: pass
    elif flag == 'Short':
        flag_rsi_short_close.append(df2.iloc[i]['RSI'])
        flag_rsi_long_open.append(df2.iloc[i]['RSI'])
        tmp_close_flag = 0
        if len(flag_rsi_short_close) == 7:
            for j in range(len(flag_rsi_short_close)-1):
                if flag_rsi_long_close[j] < flag_rsi_long_close[j+1]:
                    tmp_close_flag += 1
            if tmp_close_flag == 0:
                capital *= (position[2] / df2.iloc[i]['Close'])
                print([df2.iloc[i]['Full_Date'], 'Short Closed', df2.iloc[i]['Close'], df2.iloc[i]['RSI'], capital])
                flag = None
        try:
            if flag_rsi_long_open[1] < long_thres and flag_rsi_long_open[0] > flag_rsi_long_open[1] \
                    and flag_rsi_long_open[2] > flag_rsi_long_open[1]:
                if flag == 'Short':
                    capital *= (position[2] / df2.iloc[i]['Close'])
                    print([df2.iloc[i]['Full_Date'], 'Short Closed', df2.iloc[i]['Close'], df2.iloc[i]['RSI'], capital])
                flag = 'Long'
                position = [df2.iloc[i]['Full_Date'], 'Long', df2.iloc[i]['Close'], df2.iloc[i]['RSI']]
                print(position)
        except: pass