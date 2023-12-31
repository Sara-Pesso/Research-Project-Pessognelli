## Baye's Parameter Estimation
####################################################################################################################################################################################
## User Inputs:
file = 'case_counts_ebola2016.csv'
####################################################################################################################################################################################
####################################################################################################################################################################################


import os
dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from math import *

## Import COVID-19 Data (USA-- by county, beginning 1/22/2020)
import pandas as pd
df = pd.read_csv(file)

## Add a new column: New Confirmed Cases and New Deaths (these will represent X(1), X(2),..., X(n))
total_confirmed = list(df['Confirmed']) # The data from the CDC is strictly the total (Tn) number of conformed cases
new_confirmed = [total_confirmed[i] - total_confirmed[i-1] for i in range(1,len(total_confirmed))] # By subtracting out the previous total, we will be left with X(n)-- the number of new cases in the generation (see Eqn. on p.174 of GIP)

total_deaths = list(df['Deaths'])
new_deaths = [total_deaths[i] - total_deaths[i-1] for i in range(1,len(total_deaths))] 

## Add the as columns on to our dataframe
new_confirmed.insert(0,0)
df['New Confirmed [X(n)]'] = new_confirmed
new_deaths.insert(0,0)
df['New Deaths'] = new_deaths

## We need to determine how many "offspring" (i.e., infectees) each infected person has from one generation the the next.
## Early in a pandemic, it is safe to assume that the infectees of each infector are independent. 
generation_time = 10 # day
initial_generation = 5
generation_time = floor(generation_time)
## By independence, EX(n) = (EY(n-1))^X(n)
## Or, (new_cases gen. n+1) = (average number of infectees of gen. n)^(number of infectors in gen. n) 

Tn = []
for i in range(len(df['Day'])):
    if i ==0 :
        cases = df['Confirmed'][i]
    if i !=0:
        day2 = df['Day'][i]
        day1 = df['Day'][i-1]
        
        cases = max(df['Confirmed'][i], cases)
        
        if day2 - day1 > 1:
            while day2 - day1 > 1:
                Tn.append(cases)
                day1 += 1
        elif day2 - day1 < 1:
            Tn.append(cases)

Tn.append(max(list(df['Confirmed'])))

Tn = Tn[0::generation_time]
Xn = [Tn[i]- Tn[i-1] for i in range(1,len(Tn))]
Xn.insert(0, Tn[0])

n = [i for i in range(initial_generation,len(Xn)+initial_generation-1)] ## the 4 and 3 represent the fact that 1/22/2022 was not gen. n=1 (?)
n_copy = n.copy()
yn_1 = [Xn[i]**(1/n[i]) for i in range(len(n))]

sum_yi = sum(yn_1)
n = len(yn_1)

alpha = 1
beta = 1
print("Expected Value = R0:", (alpha+sum_yi)/(n+beta))

plot_y =  []
plot_n = []
plot_xn = []
for i in range(1, len(yn_1)):
    if yn_1[i] != 0:
        plot_y.append(yn_1[i])
        plot_n.append(n_copy[i])
        plot_xn.append(Xn[i])
plt.scatter(plot_n, plot_y)
# plt.plot(range(0,len(Tn)), list(Tn))
plt.xlabel("Generation (n = 10 days)")
plt.ylabel("Estimate of R0")
plt.title("Estimates of R0: Initial Reported Generation = 10 (WHO dataset)")
plt.show()

plt.scatter(plot_n,plot_xn)
plt.xlabel("Generation (n = 10 days)")
plt.ylabel("Newly Reported Cases (WHO)")
plt.title("Xn: Newly Reported Ebola Cases W. Africa, 2014-16 (WHO)")
plt.show()