import pandas as pd
import scipy.optimize as sc_opt
from random import uniform, seed
import matplotlib.pyplot as plt
import numpy as np

seed(0)
#### COMMON FUNCTIONS ##########################################################################
#breaking x tick labels into lines - in order to avoid overlapping on plots
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

#function computing cumulative returns
def cumul_return(returns_data, weights, rebal_freq):
    _cumul_return = [0]
    curr_weights = weights.copy()
    for i in range(returns_data.shape[0]):
        curr_weights = curr_weights * (returns_data[i,]+1)
        _cumul_return.append(np.dot(curr_weights, returns_data[i,]+1)-1)
        if rebal_freq != 0:
            if i % rebal_freq == rebal_freq - 1:
                sm = sum(curr_weights)
                curr_weights = weights*sm
    return _cumul_return

#function plotting cumulative returns + saving to csv files
def cumul_returns_plot(test_data, w_opt, train_data, strategy_no, task_no = 1):
    rets = {}
    freq_dict = {1: 'Monthly rebalancing', 3: 'Quarterly rebalancing',
                                6: 'Semi-annually rebalancing', 0:'No rebalancing'}
    for freq in (1,3,6,0):
        rets[freq_dict[freq]] = cumul_return(np.array(test_data.drop(columns=['return_end_date'])),
                                  np.array(w_opt).flatten(), freq)

    _out = pd.DataFrame(rets)
    _out['Return end date'] = [train_data['return_end_date'].iloc[-1]] + list(test_data['return_end_date'])
    _out.to_csv('Cumulative_returns_Task_' + str(task_no) + '_Strategy_' + str(strategy_no) + '.csv', index=False)

    fig, ax = plt.subplots(figsize = (16,7))
    for freq in rets:
        ax.plot([train_data['return_end_date'].iloc[-1]] + list(test_data['return_end_date']), rets[freq], label = freq)
    ax.set_title('Cumulative return in the Task %i, Strategy %i.' % (task_no, strategy_no))
    ax.tick_params(axis='x', which = 'major', labelsize = 8)
    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative return')
    ax.axhline(y = 0, color = 'black', ls = '--', lw = 0.5)
    fig.tight_layout()
    plt.legend()
    plt.savefig('Cumulative_returns_Task_' + str(task_no) + '_Strategy_' + str(strategy_no))

#function plotting weights
def weights_plot(w_opt, strategy_no, task_no = 1):
    fig, ax = plt.subplots(figsize = (16,7))
    rect = ax.bar(x=list(map(lambda x: break_line(x), w_opt.index)), height=w_opt['weight'])
    ax.set_title('Asset allocation in the Task %i, Strategy %i.' % (task_no, strategy_no))
    ax.tick_params(axis='x', which = 'major', labelsize = 8)
    ax.set_xlabel('Index')
    ax.set_ylabel('Weight')
    ax.axhline(y = 0, color = 'black', ls = '--', lw = 0.5)
    ax.bar_label(rect, padding = 2)
    fig.tight_layout()
    plt.savefig('Weights_plot_Task_' + str(task_no) + '_Strategy_' + str(strategy_no))

#risk performance measures
def risk_measures(w_opt, mi, sigma, rf, train, benchmark_train, strategy_no, task_no):
    data = pd.DataFrame()
    data['Portfolio return'] = train.drop(columns=['return_end_date']).dot(w_opt)['weight']
    data['Benchmark return'] = benchmark_train['Return']
    beta = data.cov().loc['Portfolio return', 'Benchmark return']/(data['Benchmark return'].std())**2
    exp_market_return = data['Benchmark return'].mean()
    act_return_std = (data['Portfolio return'] - data['Benchmark return']).std()
    measures = {}
    exp_return = w_opt.transpose().dot(mi).loc['weight']
    exp_std = pow(w_opt.transpose().dot(sigma.dot(w_opt)).loc['weight', 'weight'],0.5)
    measures['Sharpe'] = [(exp_return - rf)/exp_std]
    measures['Treynor'] = [(exp_return - rf)/beta]
    measures['Jensen'] = [exp_return - (rf + beta*(exp_market_return - rf))]
    measures['Information'] = [(exp_return - exp_market_return)/act_return_std]
    measures['q95'] = [data['Portfolio return'].quantile(0.05)]
    pd.DataFrame(measures).to_csv('Performance_measures_Task_' + str(task_no) +
                                  '_Strategy_' + str(strategy_no) + '.csv', index=False)
    print(measures)

##############################################################################################

#reading data with prices - note that sep and decimal characters are different than usually used (',' and '.')
prices_data = pd.read_csv('prices_tab1_ssga.csv', sep = ';', decimal = ',')
benchmark_data = pd.read_csv('benchmark_ssga.csv', sep = ';', decimal = ',')

#creating data frames with simple returns
returns_data = pd.DataFrame()
for c in prices_data.columns:
    if c == 'Dates':
        returns_data['return_end_date'] = prices_data['Dates'][1:]
    else:
        returns_data[c] = prices_data[c]/prices_data[c].shift(1) - 1

returns_benchmark = pd.DataFrame()
for c in ('Date', 'Close'):
    if c == 'Date':
        returns_benchmark['return_end_date'] = benchmark_data['Date'][1:]
    else:
        returns_benchmark['Return'] = benchmark_data[c]/benchmark_data[c].shift(1) - 1

#creating train and test sets - 80% first returns come into the train dataset
#and 20% remaining returns come into the test dataset
train, test = returns_data.iloc[0:int(returns_data.shape[0]*0.8)], \
              returns_data.iloc[int(returns_data.shape[0]*0.8):]
benchmark_train, benchmark_test = returns_benchmark.iloc[0:int(returns_benchmark.shape[0]*0.8)], \
                                  returns_benchmark.iloc[int(returns_benchmark.shape[0]*0.8):]

#computing covariance matrix and vector of mean returns for each index + specifying risk free interest rate
sigma = train.cov()
mi = train.mean(axis = 0)
rf = 0.005

# For each strategy, we perform the following steps:
# 1) Define function to minimize;
# 2) Specify constraints;
# 3) Compute performance measures;
# 4) Display graph with asset allocation;
# 5) Display graph with achieved cumulative returns.

#Strategy 1 - 60% World equities - 40% Global bonds
def f1(w, *lm):
    #lm state for lambda in the equation f(w) = min w^T mi - lambda w^T Sigma w
    #(lambda is a keyword in Python, so lm variable name is used)
    s = -w.transpose().dot(mi) + lm[0]*w.transpose().dot(sigma.dot(w))
    return s

#sum of weigths = 1
eq_lc1 = sc_opt.LinearConstraint(pd.DataFrame({'weight': [1]*(len(train.columns)-1)}).transpose(), 1, 1)
#sum of World equities weights = 0.6 & sum of Global bonds weigths = 0.4
eq_lc2 = sc_opt.LinearConstraint(pd.DataFrame({'weight': [1]*6 + [0]*4}).transpose(), 0.6, 0.6)
eq_lc3 = sc_opt.LinearConstraint(pd.DataFrame({'weight': [0]*6 + [1]*4}).transpose(), 0.4, 0.4)

min_res, w_opt = float('inf'), None
k = 0
while k <= 10:
    #initializing starting point in the optimization problems
    w0 = pd.DataFrame({'weight': [uniform(0,1) for _ in range(len(train.columns)-1)]},
                      index = list(filter(lambda x: x != 'return_end_date', train.columns)))
    res = sc_opt.minimize(f1, x0= w0, tol=pow(10,-16), args= (1,),
                          options={'maxiter':100000},
                          constraints=[eq_lc1, eq_lc2, eq_lc3], method = 'SLSQP',
                          bounds=[(0,1) for _ in range(len(returns_data.columns)-1)])

    if res.message == 'Optimization terminated successfully':
        if res.fun < min_res:
            min_res = res.fun
            w_opt = pd.DataFrame({'weight': map(lambda x: round(x, 4), res.x)},
                                 index=list(filter(lambda x: x != 'return_end_date', train.columns)))
        k += 1

cumul_returns_plot(test, w_opt, train, 1, 1)
weights_plot(w_opt, 1, 1)
risk_measures(w_opt, mi, sigma, rf, train, benchmark_train, 1, 1)

#Strategy 2 - Equal risk contribution
def f2(w):
    s = 0
    sigma_w = sigma.dot(w)
    for i in range(len(w)):
        for j in range(len(w)):
            s += (w[i]*sigma_w[i] - w[j]*sigma_w[j])**2
    return s

#sum of weigths = 1
eq_lc1 = sc_opt.LinearConstraint(pd.DataFrame({'weight': [1]*(len(train.columns)-1)}).transpose(), 1, 1)

min_res, w_opt = float('inf'), None
k = 0
while k <= 10:
    #initializing starting point in the optimization problems
    w0 = pd.DataFrame({'weight': [uniform(0,1) for _ in range(len(train.columns)-1)]},
                      index = list(filter(lambda x: x != 'return_end_date', train.columns)))
    res = sc_opt.minimize(f2, x0= w0, tol=pow(10,-16),
                          options={'maxiter':100000},
                          constraints=[eq_lc1], method = 'SLSQP',
                          bounds=[(0,1) for _ in range(len(returns_data.columns)-1)])

    if res.message == 'Optimization terminated successfully':
        if res.fun < min_res:
            min_res = res.fun
            w_opt = pd.DataFrame({'weight': map(lambda x: round(x, 4), res.x)},
                                 index=list(filter(lambda x: x != 'return_end_date', train.columns)))
        #print(res)
        k += 1

cumul_returns_plot(test, w_opt, train, 2, 1)
weights_plot(w_opt, 2, 1)
risk_measures(w_opt, mi, sigma, rf, train, benchmark_train, 2, 1)

#Strategy 3 - Minimum variance strategy
def f3(w):
    s = w.transpose().dot(sigma.dot(w))
    return s

#sum of weigths = 1
eq_lc1 = sc_opt.LinearConstraint(pd.DataFrame({'weight': [1]*(len(train.columns)-1)}).transpose(), 1, 1)

min_res, w_opt = float('inf'), None
k = 0
while k <= 10:
    #initializing starting point in the optimization problems
    w0 = pd.DataFrame({'weight': [uniform(0,1) for _ in range(len(train.columns)-1)]},
                      index = list(filter(lambda x: x != 'return_end_date', train.columns)))
    res = sc_opt.minimize(f3, x0= w0, tol=pow(10,-16),
                          options={'maxiter':100000},
                          constraints=[eq_lc1], method = 'SLSQP',
                          bounds=[(0,1) for _ in range(len(returns_data.columns)-1)])

    if res.message == 'Optimization terminated successfully':
        if res.fun < min_res:
            min_res = res.fun
            w_opt = pd.DataFrame({'weight': map(lambda x: round(x, 4), res.x)},
                                 index=list(filter(lambda x: x != 'return_end_date', train.columns)))
        #print(res)
        k += 1

cumul_returns_plot(test, w_opt, train, 3, 1)
weights_plot(w_opt, 3, 1)
risk_measures(w_opt, mi, sigma, rf, train, benchmark_train, 3, 1)

#Strategy 4 - Equal weight
#There is nothing to optimize, we simply assume equal weights.
w_opt = pd.DataFrame({'weight': [round(1/(len(train.columns)-1), 4) for _ in range(len(train.columns)-1)]},
                     index=list(filter(lambda x: x != 'return_end_date', train.columns)))

cumul_returns_plot(test, w_opt, train, 4, 1)
weights_plot(w_opt, 4, 1)
risk_measures(w_opt, mi, sigma, rf, train, benchmark_train, 4, 1)