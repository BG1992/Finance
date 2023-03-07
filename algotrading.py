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

df.sort_values(by='Full_Date', inplace=True, ascending = True)
df.reset_index(inplace=True, drop=True)
print(df.shape)
df = df[df['Delay'] == dt.timedelta(minutes = -1)][['Full_Date', 'Close']]
print(df.shape)

#df = df.iloc[:1000]
X = []
y = []
df_train, df_test = df.iloc[:df.shape[0]*4//5].reset_index(drop=True), \
                    df.iloc[df.shape[0]*4//5:].reset_index(drop=True)
for i in range(120, df_train.shape[0]-101):
    if i % 1000 == 0: print(i, df_train.shape[0], sum(y))
    tb = []
    for k in (120, 60, 15, 5):
        df_tmp = df_train.iloc[i-k:i].reset_index(drop=True)
        model = linear_model.LinearRegression().fit(df_tmp.index.factorize()[0].reshape(-1, 1), df_tmp['Close'])
        sigma = df_tmp['Close'].std()
        tb.extend([model.coef_[0], model.intercept_, sigma])
    res = 0
    o = df_train.iloc[i]['Close']
    closed = False
    for j in range(i+1, i + 101):
        if df_train.iloc[j]['Close'] - df_train.iloc[i]['Close'] < -40 * pow(10, -5):
            c = df_train.iloc[j]['Close']
            res = 0
            closed = True
            break
        if df_train.iloc[j]['Close'] - df_train.iloc[i]['Close'] > 25 * pow(10, -5):
            c = df_train.iloc[j]['Close']
            res = 1
            closed = True
    if not closed:
        c = df_train.iloc[i+100]['Close']
        if c > o+5: res = 1
    X.append(tb)
    y.append(res)

print(sum(y), len(y))
lm = linear_model.LogisticRegression(class_weight='balanced').fit(X, y)
print(lm.score(X, y))
X, y = [], []
capital = 1000
pos_ct, ct = 0, 0
i = 120
while i < df_test.shape[0]-101:
    if i % 1000 == 0: print(i, df_test.shape[0])
    tb = []
    for k in (120, 60, 15, 5):
        df_tmp = df_test.iloc[i-k:i].reset_index(drop=True)
        model = linear_model.LinearRegression().fit(df_tmp.index.factorize()[0].reshape(-1, 1), df_tmp['Close'])
        sigma = df_tmp['Close'].std()
        tb.extend([model.coef_[0], model.intercept_, sigma])
    res = 0
    o = df_test.iloc[i]['Close']
    closed = False
    for j in range(i+1, i + 101):
        if df_test.iloc[j]['Close'] - df_test.iloc[i]['Close'] < -40 * pow(10, -5):
            c = df_test.iloc[j]['Close']
            res = 0
            closed = True
            break
        if df_test.iloc[j]['Close'] - df_test.iloc[i]['Close'] > 25 * pow(10, -5):
            c = df_test.iloc[j]['Close']
            res = 1
            closed = True
            break
    if not closed:
        c = df_test.iloc[i+100]['Close']
        if c > o+5: res = 1
    if lm.predict([tb]) == 1:
        capital *= (c/o)
        if c > o+5: pos_ct += 1
        ct += 1
        print(i, capital)
        i = j+1
    X.append(tb)
    y.append(res)

if len(X) > 0:
    y_pred = lm.predict(np.array(X))
    print(y_pred.sum())
    print(lm.score(X, y))
    print(capital)
    print(pos_ct, ct, pos_ct/ct)