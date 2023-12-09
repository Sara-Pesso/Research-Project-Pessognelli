### SIR Model Example
# The SIR model is a simple ODE model used for predicting the spread of disease in the later stages of an outbreak.
# This particular example uses Berks County, PA, but the inputs can be adjusted for any county/country data of suitable form.
##############################################################################################################################################################################
# User Inputs:
total_population = 421017 # Population of Berks County, PA in 2020
peak_infected = 5044 # Max number of infected at one time in Berks
b = 0.178 # SIR parameters
k = 0.015
generation_time = 5 # 5 days is accepted infectious period of COVID 19
total_time = 250 #simulation time
##############################################################################################################################################################################
##############################################################################################################################################################################

import matplotlib.pyplot as plt
import pandas as pd
import os
dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)

df = pd.read_csv('Berks.csv')

s0 = total_population
i0 = peak_infected/(7.4*(10**5))
r0 = i0

time_step = 1 

st = [s0/s0]
it = [i0/s0]
rt = [r0/s0]

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
print(it[1])


## Add a new column: New Confirmed Cases and New Deaths (these will represent X(1), X(2),..., X(n))
total_confirmed = list(df['Confirmed']) # The data from the CDC is strictly the total (Tn) number of conformed cases
Tn = list(df['Confirmed'])

Tn = Tn[0::generation_time]

Tn = [t/s0 for t in Tn]

# plot lines 
plt.plot(times, st, label = "S(t)")
plt.plot(times, it, label = "I(t)") 
plt.plot(times, rt, label = "R(t)") 
plt.title("Example SIR Data")
plt.xlabel("Time (n)")
plt.ylabel("Proportion of Total Population")
plt.legend() 
plt.show()