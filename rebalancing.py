import pandas as pd
import numpy as np
import os
import datetime as dt
import cvxopt

cvxopt.solvers.options['show_progress'] = False
folder_path = r'C:\Users\barto\Documents\nc_stocks'

dfs = []
for f in list(os.listdir(folder_path)):
    try:
        _df = pd.read_csv(os.path.join(folder_path, f))
        _df.rename(columns = {'<CLOSE>': f.replace('.txt', ''), '<DATE>': 'Date'}, inplace = True)
        _df = _df[['Date', f.replace('.txt', '')]]
        _df['Date'] = pd.to_datetime(_df['Date'], format='%Y%m%d')
        _df.set_index(keys=['Date'], drop = True, inplace = True)
        dfs.append(_df)
    except: pass

df = pd.concat(dfs, axis=1, join='outer')
dfs = None
df = df[(df.index > dt.datetime(2017, 1, 1)) & (df.index <= dt.datetime(2022, 3, 31))]

cols_to_drop = []
for col in df.columns:
    if df[col].isna().sum()/df[col].shape[0] > 0.1: cols_to_drop.append(col)

df.drop(columns=cols_to_drop, inplace = True)
df.fillna(method='bfill', inplace= True)

df_returns = df.copy()
for col in df.columns:
    df_returns[col] = df[col]/df[col].shift(1)-1

prev_day = dt.datetime(2017, 3, 29)
df_start = df_returns[(df_returns.index > dt.datetime(2017, 1, 2)) & (df_returns.index < prev_day)]

alfa = -0.1
fee = 0.39/100
mean_length = 5
mi_close_lb = 1.1
step_len = 10
capital = 1 - fee

Q = cvxopt.matrix(df_start.cov().values*alfa)
#r = cvxopt.matrix(-df_start.iloc[-mean_length:].mean())
r = cvxopt.matrix([0.0]*df_start.shape[1])
A = cvxopt.matrix(np.ones(shape=(1, df_start.shape[1])))
b = cvxopt.matrix([capital])
G = cvxopt.matrix((-1)*np.eye(df_start.shape[1]))
h = cvxopt.matrix([0.0]*df_start.shape[1])
sol = cvxopt.solvers.qp(Q, r, G, h, A, b)

curr_day = prev_day + dt.timedelta(days=step_len)
while curr_day not in df.index:
    curr_day += dt.timedelta(days=1)

capitals = []

while curr_day <= dt.datetime(2022, 3, 15):
    w = np.array(sol['x']).transpose().squeeze()
    ret = (df[df.index == curr_day].values/df[df.index == prev_day].values).transpose().squeeze()
    indices_to_close = np.where(ret > mi_close_lb)[0]
    profit = np.dot(w[indices_to_close], df[df.index == curr_day].values.squeeze()[indices_to_close]/
                               df[df.index == prev_day].values.squeeze()[indices_to_close] - 1)
    capital = np.dot(w, ret)
    capital -= fee*capital
    print(prev_day, curr_day, indices_to_close.shape[0], profit, capital)
    # capital = np.dot(w, ret)
    # print(prev_day, curr_day, capital)
    capitals.append(capital)
    df_curr = df_returns[(df_returns.index < prev_day - dt.timedelta(days=15)) & (df_returns.index < curr_day)]
    Q = cvxopt.matrix(df_curr.cov().values * alfa)
    #r = cvxopt.matrix(-df_curr.iloc[-mean_length:].mean().values)
    r = cvxopt.matrix([0.0] * df_start.shape[1])
    _A = np.zeros(shape=(indices_to_close.shape[0], df_curr.shape[1]))
    curr_row = 0
    for ind in indices_to_close:
        _A[curr_row, ind] = 1
        curr_row += 1
    A = cvxopt.matrix(np.vstack((np.ones(shape=(1, df_curr.shape[1])), _A)))
    b = cvxopt.matrix([capital] + [0.0]*indices_to_close.shape[0])
    # A = cvxopt.matrix(np.ones(shape=(1, df_curr.shape[1])))
    # b = cvxopt.matrix([capital])
    G = cvxopt.matrix((-1) * np.eye(df_curr.shape[1]))
    h = cvxopt.matrix([0.0] * df_curr.shape[1])
    sol = cvxopt.solvers.qp(Q, r, G, h, A, b)

    prev_day = curr_day
    curr_day += dt.timedelta(days=step_len)
    while curr_day not in df.index:
        curr_day += dt.timedelta(days=1)

print(np.dot(w, ret))
capitals = np.array(capitals)
print(min(capitals), np.percentile(capitals, 25), np.percentile(capitals, 50), np.percentile(capitals, 75), max(capitals))