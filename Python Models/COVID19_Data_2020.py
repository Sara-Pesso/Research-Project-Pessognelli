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
df = pd.read_csv(r'E:\Research Project\Python Models\COVID19_Data_2020\LosAngeles.csv')

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

### Paramter Estimation Using Baye's Theorem 
## We need to determine how many "offspring" (i.e., infectees) each infected person has from one generation the the next.
## Early in a pandemic, it is safe to assume that the infectees of each infector are independent. 
generation_time = 5
generation_time = floor(generation_time)
## By independence, EX(n) = (EY(n-1))^X(n)
## Or, (new_cases gen. n+1) = (average number of infectees of gen. n)^(number of infectors in gen. n) 

Tn = list(df['Confirmed'])
i = 0
while i < len(Tn):
    if Tn[i] <= 0:
        del Tn[i]
    else:
        i+=1

Tn = Tn[0::generation_time]
Xn = [Tn[i]- Tn[i-1] for i in range(1,len(Tn))]
Xn.insert(0, Tn[0])

n = [i for i in range(1,len(Xn)+0)] ## the 4 and 3 represent the fact that 1/22/2022 was not gen. n=1 (?)
yn_1 = [Xn[i]**(1/n[i]) for i in range(len(n))]

sum_yi = sum(yn_1)
n = len(yn_1)

alpha = 1
beta = 1
print("Expected Value = R0:", (alpha+sum_yi)/(n+beta))
print(yn_1)
plt.plot(range(n), yn_1)
plt.plot(range(0,len(list(df['Confirmed']))), list(df['Confirmed']))
plt.show()


