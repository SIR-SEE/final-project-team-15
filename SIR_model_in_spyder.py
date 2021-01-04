#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 15:36:31 2021

@author: edvindannas
"""

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt



# describe the model
def deriv(y, t, N, beta, gamma, delta, alpha, mu, kappa):
    S, E, I, R, D, V = y
    vacc = 200 # introduction day for vaccine
    
    k = 1 if vacc < t < vacc + 1/kappa else 0 
    # variable activating vaccination until whole population vaccinated
    
    dSdt = -beta(t) * I * S / N + mu * R - S * kappa * k
    dVdt = (N-D) * kappa * k 
    # random vaccination gives proportional distribution between S, E, I and R, leaving out D.
    
    dEdt = beta(t) * I * S / N - delta * E - E * kappa * k
    dIdt = delta * E - (1 - alpha) * gamma * I - alpha * I - I * kappa * k
    dRdt = (1 - alpha) * gamma * I - mu * R - R * kappa * k
    dDdt = alpha * I
              
    return dSdt, dEdt, dIdt, dRdt, dDdt, dVdt

def R_0(t): # behavioral dependent
    return 12 if t < L else 2 # the value of R0 changes if a lockdown is introduced

def beta(t): # rate of spread (number of exposed per individual and day)
    return R_0(t) * gamma

# describe the parameters
N =  7000000                 # population
Days_infection = 8           # duration of infection
Days_incubation = 8          # incubation time
delta = 1.0/Days_incubation  # incubation rate
alpha = 1/60                 # death percentage
gamma = 1/Days_infection     # recovering rate
mu = 1/1000                  # 1/duration of immunity
L = 60                       # day of lockdown
kappa = 1/200                # vaccination percentage per day of population

              
S0, E0, I0, R0, D0, V0 = (N-1), 1, 0, 0, 0, 0 # initial conditions: one exposed, rest susceptible



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