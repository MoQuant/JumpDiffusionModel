import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd

def dWT():
    return rd.randint(-5, 5)/100.0

def Stochastic(ror, dt):
    # kappa, theta, sigma
    vol = []
    window = 30

    for t in range(window, len(ror)):
        hold = ror[t-window:t]
        vol.append(np.std(hold)**2)

    theta = np.mean(vol)
    sigma = np.std(vol)

    M = len(vol)
    top = np.sum([(vol[i] - theta)*(vol[i-1] - theta) for i in range(1, M)])
    bot = np.sum([pow(vol[i] - theta, 2) for i in range(M)])

    kappa = -np.log(top/bot)/dt

    drift = np.mean(ror)
    variance = np.std(ror)**2

    return drift, variance, kappa, theta, sigma


data = pd.read_csv('SPY.csv')[::-1]

data = data[(data['date'] >= '2023-10-20')]

close = data['adjClose'].values
ror = close[1:]/close[:-1] - 1.0

stockPrice = close[-1]

T = 30.0/365.0
N = 500
dt = T/N

drift, varz, kappa, theta, sigma = Stochastic(ror, dt)

P = 100

lb = [0.5]

Paths = []
v0 = varz
for paths in range(P):
    S0 = stockPrice
    v0 = varz
    Wt = []
    for i in range(N):
        dw = dWT()
        Wt.append(dw)
        v0 += max(kappa*(theta - v0)*dt + sigma*np.sqrt(v0)*dw, 0)

        jump = 0
        rs = rd.random()
        if rs < np.mean(lb)*dt:
            jump = np.exp(np.mean(lb) + np.std(lb)*rs) - 1.0
            lb.append(jump)
            
        S0 = S0*np.exp(((drift + 0.5*v0)*dt + np.sqrt(v0)*dw)*(1 + jump))
        
    Paths.append(S0)

log_returns = [np.log(p/stockPrice) for p in Paths]

slog = list(sorted(log_returns))

alpha = 0.05

VaR = slog[int(alpha*len(slog))]
CVaR = np.sum([r for r in log_returns if r <= VaR])/np.sum([1 for r in log_returns if r <= VaR])

print("SPYDER ETF Analysis")
print(f'Value At Risk w/ Alpha = {alpha} is {VaR}')
print(f'Conditional Value At Risk w/ Alpha = {alpha} is {CVaR}')

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)

ax.hist(log_returns, bins=15, color='cyan', edgecolor='black')
ax.axvline(VaR, color='red', linestyle='--', label=f'VaR @ {alpha}')
ax.axvline(CVaR, color='orange', linestyle='--', label=f'CVaR @ {alpha}')
ax.set_title("SPY Distribution Graph")
ax.set_xlabel("Returns")
ax.set_ylabel("Frequency")
ax.legend()

plt.show()







