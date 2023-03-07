import pandas as pd
import scipy.optimize as sc_opt
from random import uniform
import matplotlib.pyplot as plt
import numpy as np

prices_data = pd.read_csv('returns_tab1_ssga.csv', sep = ';', decimal = ',')

returns_data = pd.DataFrame()
for c in prices_data.columns:
    if c == 'Dates':
        returns_data['return_end_date'] = prices_data['Dates'][1:]
    else:
        returns_data[c] = prices_data[c]/prices_data[c].shift(1) - 1

train, test = returns_data.iloc[0:int(returns_data.shape[0]*0.8)], returns_data.iloc[int(returns_data.shape[0]*0.8):]
sigma = train.cov()
mi = train.mean(axis = 0)

def break_line(tick, max_length=8):
    _tick = ''
    k = 0
    for c in tick:
        if c == ' ' and k > max_length:
            _tick += '\n'
            k = 0
        else:
            k += 1
            _tick += c
    return _tick

def f(w):
    #print(w.transpose().dot(sigma.dot(w)))
    return w.transpose().dot(sigma.dot(w))

w0 = pd.DataFrame({'weight': [uniform(0,1) for _ in range(len(train.columns)-1)]},
                  index = list(filter(lambda x: x != 'return_end_date', train.columns)))

eq_lc = sc_opt.LinearConstraint(pd.DataFrame({'weight': [1]*(len(train.columns)-1)}).transpose(), 1, 1)
res = sc_opt.minimize(f, x0= w0, tol=pow(10,-16), options={'maxiter':100000}, constraints=[eq_lc])

w_opt = pd.DataFrame({'weight': map(lambda x: round(x,3), res.x)},
                  index = list(filter(lambda x: x != 'return_end_date', train.columns)))

print(w_opt)

# fig, ax = plt.subplots()
# rect = ax.bar(x=list(map(lambda x: break_line(x), w_opt.index)), height=w_opt['weight'])
# ax.set_title('Asset allocation in the Task 1, Strategy 3.')
# ax.tick_params(axis='x', which = 'major', labelsize = 8)
# ax.set_xlabel('Index')
# ax.set_ylabel('Weight')
# ax.axhline(y = 0, color = 'black', ls = '--', lw = 0.5)
# ax.bar_label(rect, padding = 2)
# fig.tight_layout()
# plt.show()

def cumul_return(returns_data, weights, rebal_freq):
    _cumul_return = [1]
    curr_weights = weights.copy()
    for i in range(returns_data.shape[0]):
        curr_weights = curr_weights * (returns_data[i,]+1)
        _cumul_return.append(np.dot(curr_weights, returns_data[i,]+1))
        if rebal_freq != 0:
            if i % rebal_freq == rebal_freq - 1:
                sm = sum(curr_weights)
                curr_weights = weights*sm
    return _cumul_return

rets = {}
for freq in (1,3,6,0):
    rets[freq] = cumul_return(np.array(test.drop(columns=['return_end_date'])), np.array(w_opt).flatten(), freq)

fig, ax = plt.subplots()
for freq in rets:
    ax.plot([train['return_end_date'].iloc[-1]] + list(test['return_end_date']), rets[freq], label = freq)
ax.set_title('Cumulative return in the Task 1, Strategy 3.')
ax.tick_params(axis='x', which = 'major', labelsize = 8)
ax.set_xticks(ax.get_xticks()[::2])
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative return')
ax.axhline(y = 1, color = 'black', ls = '--', lw = 0.5)
fig.tight_layout()
plt.legend()
plt.show()

rf = 0.001
exp_return = w_opt.dot(mi)
exp_std = w_opt.dot(sigma.dot(w_opt.transpose()))
measures = {}
#measures['Sharpe'] = (exp_return - rf)/exp_std
#measures['Sortino'] =