import os
dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from math import *

## Import COVID-19 Data (day_waise.csv-- world wide infection tracker)
import pandas as pd
df = pd.read_csv(r'E:\Research Project\Python Models\COVID19_Data_2020\day_wise.csv')

### Paramter Estimation Using Baye's Theorem 
## We need to determine how many "offspring" (i.e., infectees) each infected person has from one generation the the next.
## Early in a pandemic, it is safe to assume that the infectees of each infector are independent. 
generation_time = 5
generation_time = floor(generation_time)
## By independence, EX(n) = (EY(n-1))^X(n)
## Or, (new_cases gen. n+1) = (average number of infectees of gen. n)^(number of infectors in gen. n) 

Tn = list(df['Confirmed'])
Tn = Tn[0::generation_time]

Xn = [Tn[i]- Tn[i-1] for i in range(1,len(Tn))]
Xn.insert(0, Tn[0])
print(Xn)

n = [i for i in range(4,len(Xn)+3)] ## the 4 and 3 represent the fact that 1/22/2022 was not gen. n=1
yn_1 = [Xn[i]**(1/n[i]) for i in range(len(n))]

# print(sum(yn_1))
# print(len(yn_1))

sum_yi = sum(yn_1)
n = len(yn_1)

alpha = 1
beta = 1
print("Expected Value = R0:", (alpha+sum_yi)/(n+beta))