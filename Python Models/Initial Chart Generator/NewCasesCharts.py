import os
dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from math import *

## Import COVID-19 Data (USA-- by county, beginning 1/22/2020)
berks_df = pd.read_csv(r'Berks.csv')
LA_df = pd.read_csv(r'LosAngeles.csv')
NYC_df = pd.read_csv(r'Manhattan.csv')
Philly_df = pd.read_csv(r'Philadelphia.csv')

## Tn = Total Number of Cases, by day
def gen0_finder(Tn):
    i = 0
    while i < len(Tn):
        if Tn[i] <= 0:
            del Tn[i]
        else:
            i+=1
    return Tn

berks_Tn = list(berks_df['Confirmed'])
berks_Tn = gen0_finder(berks_Tn)

LA_Tn = list(LA_df['Confirmed'])
LA_Tn = gen0_finder(LA_Tn)

NYC_Tn = list(NYC_df['Confirmed'])
NYC_Tn = gen0_finder(NYC_Tn)

Philly_Tn = list(Philly_df['Confirmed'])
Philly_Tn = gen0_finder(Philly_Tn)

## check
plt.plot(range(len(list(berks_df["Confirmed"]))),list(berks_df["Confirmed"]))
plt.show()

plt.plot(range(len(list(berks_df["Confirmed"]))),list(LA_df["Confirmed"]))
plt.show()

plt.plot(range(len(list(berks_df["Confirmed"]))),list(NYC_df["Confirmed"]))
plt.show()

plt.plot(range(len(list(berks_df["Confirmed"]))),list(Philly_df["Confirmed"]))
plt.show()
###
generation_time = 5
generation_time = floor(generation_time)
berks_Tn = berks_Tn[0::generation_time]
LA_Tn = LA_Tn[0::generation_time]
NYC_Tn = NYC_Tn[0::generation_time]
Philly_Tn = Philly_Tn[0::generation_time]

## Xn = total number of new cases, by day
berks_Xn = [berks_Tn[i]- berks_Tn[i-1] for i in range(1,len(berks_Tn))]
berks_Xn.insert(0, berks_Tn[0])

LA_Xn = [LA_Tn[i]- LA_Tn[i-1] for i in range(1,len(LA_Tn))]
LA_Xn.insert(0, LA_Tn[0])

NYC_Xn = [NYC_Tn[i]- NYC_Tn[i-1] for i in range(1,len(NYC_Tn))]
NYC_Xn.insert(0, NYC_Tn[0])

Philly_Xn = [Philly_Tn[i]- Philly_Tn[i-1] for i in range(1,len(Philly_Tn))]
Philly_Xn.insert(0, Philly_Tn[0])

## Plotting Xn: Plotting number of new cases by day
## Big cities: Philly, NYC, and LA
# days = [i for i in range(0,len(berks_Xn))]

plt.scatter([i for i in range(0, len(LA_Xn))], LA_Xn)
plt.scatter([i for i in range(0, len(NYC_Xn))], NYC_Xn)
plt.scatter([i for i in range(0, len(Philly_Xn))], Philly_Xn)
plt.title("New Cases in Major Cities, by generation")
plt.xlabel("Generation")
plt.ylabel("Number of New Cases")
plt.legend(["Los Angeles", "Manhattan", "Philadelphia"])
plt.show()
# plt.savefig("newcases_cities.png")

## Berks county, PA
plt.close()
# plt.subplot(2,2,1)
berks_gens = [i for i in range(0, len(berks_Xn))]
plt.scatter(berks_gens, berks_Xn)
plt.title("Berks, PA")
plt.xlabel("Generation")
plt.ylabel("Number of New Cases")
plt.legend(["Berks, PA"])
plt.show()
# plt.savefig("newcases_berks.png")

## Philly
# plt.subplot(2,2,2)
plt.scatter([i for i in range(0, len(Philly_Xn))], Philly_Xn)
plt.title("Philadelphia")
plt.xlabel("Generation")
plt.ylabel("Number of New Cases")
# plt.legend(["Philadelphia"])
plt.show()
# plt.savefig("newcases_philly.png")

## NYC
# plt.subplot(2,2,3)
plt.scatter([i for i in range(0, len(NYC_Xn))], NYC_Xn)
plt.title("Manhattan")
plt.xlabel("Generation")
plt.ylabel("Number of New Cases")
# plt.legend(["Manhattan"])
plt.show()
# plt.savefig("newcases_nyc.png")

## LA
# plt.subplot(2,2,4)
plt.scatter([i for i in range(0, len(LA_Xn))], LA_Xn)
plt.title("Los Angeles")
plt.xlabel("Generation")
plt.ylabel("Number of New Cases")
# plt.legend(["Los Angeles"])
# plt.show()
# plt.savefig("newcases_LA.png")

# plt.savefig("all_newcases_5days.png")
plt.show()