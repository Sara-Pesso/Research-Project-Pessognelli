s0 = 8*10**9
i0 = 1
r0 = 0

total_time = 250
time_step = 1 

st = [s0/s0]
it = [i0/s0]
rt = [r0/s0]
b = 0.2
k = 0.04
## Traditional SIR Model

for time in range(0, total_time, time_step):
    t = int(time/time_step)

    ds = -b*st[t]*it[t]
    
    s1 = st[t] + ds
    st.append(s1)

    di = b*st[t]*it[t] - k*it[t]
    i1 = it[t] + di
    it.append(i1)

    dr = k*it[t]
    r1 = rt[t] + dr
    rt.append(r1)

## Quick Plot 
times = [t for t in range(0, total_time+time_step, time_step)]

import matplotlib.pyplot as plt 
import pandas as pd
df = pd.read_csv(r'E:\Research Project\Python Models\COVID19_Data_2020\World Wide.csv')

total_confirmed = list(df['Confirmed']) 
Tn = list(df['Confirmed'])
generation_time = 5
Tn = Tn[0::generation_time]

Tn = [t/s0 for t in Tn]

# plot lines 
plt.plot(times, st, label = "S(t)")
plt.plot(times, it, label = "I(t)") 
plt.plot(times, rt, label = "R(t)") 
plt.title("Example SIR Data")
plt.xlabel("Time (n)")
plt.ylabel("Proportion")
plt.legend() 
plt.show()

# print(st[1], it[1], rt[1])

### SIR as a Probabilistic Model for the early stages of a epidemic/ pandemic. 
## https://staff.math.su.se/daniel.ahlberg/notes-epidemics.pdf
N = 320000000 # Roughly the population of the US
St = [N-1] #susceptible
It = [1] # Infected
Rt = [0] # recovered

from scipy.stats import expon, poisson
import numpy as np
lambda_ = 0.5 # rate-- the average number of occurences, per unit time, for which infection occurs. 

gamma = 0.2 ## GUESS: 5 (gamma = 1/5) infectious interactions per day
t_interaction = expon.rvs(scale = 1/gamma, size = 10000) #probabilistic variable. 
mean_interactions = np.mean(t_interaction)
Y = np.random.poisson(lam = lambda_*mean_interactions, size = 10000) # Number of infectious contacts of arbitrary infected individual
print (np.mean(Y))
# #Where t_interaction if the lenght of the interaction
# T = #interval an individual remains infectious
# #Therefore
# R0 = EY = E[E(Y|T)] = E[lambda_T] = lambda_/gamma #R0 = the reproductive number
# ## Remains accurate as long as the number of infectious contacts (Y) < sqrt(N)
# $$$ I would like to use this in the deterministic version of this model. 



