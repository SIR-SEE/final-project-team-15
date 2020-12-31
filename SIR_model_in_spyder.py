# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 15:47:37 2020

@author: Tage
"""

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt



# describe the model
def deriv(y, t, N, beta, gamma, delta, alpha, mu, kappa):
    S, E, I, R, D, V = y
    vacc = 200 #introduction day for vaccine
    
    if t < vacc:
        dSdt = -beta(t) * S * I / N +mu*R
        dVdt = 0
        
        dEdt = beta(t) * S * I / N - gamma * E #- mu*E
        dIdt = delta * E - gamma * I  - alpha * I  #- mu*I
        dRdt = gamma * I - mu* R
        dDdt = alpha * I
    elif t >= vacc:
        dSdt = -beta(t) * S * I / N +mu*R - S*kappa
        dVdt = S * kappa
        
        dEdt = beta(t) * S * I / N - gamma * E #- mu*E
        dIdt = delta * E - gamma * I  - alpha * I  #- mu*I
        dRdt = gamma * I - mu* R
        dDdt = alpha * I
        
    return dSdt, dEdt, dIdt, dRdt, dDdt, dVdt

def R_0(t):
    return 12 if t < L else 2 #the value of R0 dependent on if a lockdown is introduced

def beta(t):
    return R_0(t) * gamma #beta dependent on R0 and our behaviour

# describe the parameters
L = 60 #day of lockdown
N =  7000000           # population
Days = 8
delta = 1.0/Days
alpha = 1/60 #death percentage
gamma=1/Days #recovering per day
mu = 1/1000 #vaining immunity
kappa = 1/200 #vaccination percentage per day of population

              
S0, E0, I0, R0, D0, V0 = (N-1), 1, 0, 0, 0, 0 # initial conditions: one infected, rest susceptible



t = np.linspace(0, 365, 700) # Grid of time points (in days)
y0 = S0, E0, I0, R0, D0, V0 # Initial conditions vector

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, delta, alpha, mu, kappa))
S, E, I, R, D, V = ret.T



def plotsir(t, S, E, I, R, D, V):
  f, ax = plt.subplots(1,1,figsize=(10,4))
  ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
  ax.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
  ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
  ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
  ax.plot(t, D, 'k', alpha=0.7, linewidth=2, label='Dead')
  ax.plot(t, V, 'm', alpha=0.7, linewidth=2, label='Vaccinated')
  ax.set_xlabel('Time (days)')

  ax.yaxis.set_tick_params(length=0)
  ax.xaxis.set_tick_params(length=0)
  ax.grid(b=True, which='major', c='w', lw=2, ls='-')
  legend = ax.legend()
  legend.get_frame().set_alpha(0.5)
  for spine in ('top', 'right', 'bottom', 'left'):
      ax.spines[spine].set_visible(False)
  plt.savefig("Plot.png")
  plt.show();
  
plotsir(t, S, E, I, R, D, V)

#test to see that the population has stayed the same 
tmax = -1
poptest = S[tmax] + E[tmax] + I[tmax] + R[tmax] + D[tmax] + V[tmax]
print(poptest)