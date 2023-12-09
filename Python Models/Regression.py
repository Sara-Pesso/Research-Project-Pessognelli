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

county = "LosAngeles"
df = pd.read_csv(r'E:\Research Project\Python Models\COVID19_Data_2020\%s.csv' % county)

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


### Parameter Estimation Using Regression. 
## Now, we will linearize the new confirmed cases data.

generation_time = 5
generation_time = floor(generation_time)

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
n = [i for i in range(1,len(Xn))]

LnXn = list(np.log(Xn))
day = list(np.linspace(0, len(LnXn)-1, num = len(LnXn), endpoint = True))

i = 0
while i < len(LnXn):
    if LnXn[i] == -inf:
        del LnXn[i]
        del day[i]
    else:
        i+=1  

print(LnXn)
print(day)

#Plot Xn vs n
plt.scatter(day, LnXn)

## Best Linear Predictor (Theorem 2.5.2 in GIP)
## L(X) = alpha + beta*X
x_bar = np.mean(day)
y_bar = np.mean(LnXn)
var_x = np.var(day)
var_y = np.var(LnXn)
cov_xy = np.cov(day, LnXn)[0][1]

alpha = y_bar - (cov_xy/var_x)*x_bar
beta = cov_xy/var_x

## and thus our BLP is
def BLP(x):
    return y_bar + (cov_xy/var_x)*(x - x_bar)

## Print the estimated R0
slope_m = cov_xy/var_x
R0_hat = exp(slope_m)

print("Estimated R0:", R0_hat)

# Plot the BLP line !
plt.plot(day, BLP(day), label = "R0 = %.4f" % R0_hat)
plt.xlabel("Generation, n")
plt.ylabel("ln(Xn)")
plt.title("Linearized Regression: %s" % county)
plt.legend(loc = "lower right")
plt.show()




