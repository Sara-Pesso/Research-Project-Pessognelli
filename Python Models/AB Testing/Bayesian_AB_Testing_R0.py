## Bayesian A/B Testing for the R0 (Re) of COVID in various US counties in 2020.
from matplotlib import *
from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
from math import *
import pymc as pm
import os
dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)
## Import COVID-19 Data (USA-- by county, beginning 1/22/2020)
import pandas as pd
file = r'World Wide.csv'
df = pd.read_csv(file)
county = file.split("\\")[-1]
county = county.split(".")[0]
## Add a new column: New Confirmed Cases and New Deaths (these will represent X(1), X(2),..., X(n))
total_confirmed = list(df['Confirmed']) # The data from the CDC is strictly the total (Tn) number of conformed cases
new_confirmed = [total_confirmed[i] - total_confirmed[i-1] for i in range(1,len(total_confirmed))] # By subtracting out the previous total, we will be left with X(n)-- the number of new cases in the generation (see Eqn. on p.174 of GIP)

### Paramter Estimation Using Baye's Theorem 
## We need to determine how many "offspring" (i.e., infectees) each infected person has from one generation the the next.
## Early in a pandemic, it is safe to assume that the infectees of each infector are independent. 
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

#Spot Estimates of R0
R0_hat =[]
span = 1 # number of generations to average over
for i in range(len(LnXn)-span):
    r0_hat = (LnXn[i+span]-LnXn[i])/span # This will always be 1 generation
    r0_hat = exp(r0_hat)
    R0_hat.append(r0_hat)

gens = np.linspace(0, len(R0_hat[1:])-1, len(R0_hat[1:]))
plt.subplot(2, 2, 1)
plt.scatter(gens, R0_hat[1:])
plt.xlabel("Time (generations, n = 5 days)")
plt.ylabel("Estimated R0")
plt.title("Estimates for %s" % county)

### Create the PYMC3 model
R0_hat = R0_hat[1:]
with pm.Model() as model:
    alpha = 1.0/np.mean(R0_hat)
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    # lambda_1 = pm.Uniform("lambda_1", lower = 0, upper = 10)
    # lambda_2 = pm.Uniform("lambda_2", lower = 0, upper = 10)
    tau = pm.DiscreteUniform("tau", lower=0, upper = len(R0_hat)-1)

## Creating the randomly generated switch points in the R0 estimate data
with model:
    idx = np.arange(len(R0_hat)) # Index
    lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)

with model:
    observation = pm.Poisson("obs", lambda_, observed=R0_hat)

## MCMC
with model:
    step = pm.Metropolis()
    trace = pm.sample(50000, tune=5000, cores = 1, step=step, return_inferencedata=False)

lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']
tau_samples = trace['tau']

figsize(12.5, 10)
#histogram of the samples:

plt.subplot(2, 2, 2)
plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="mean = %.4f, var = %.4f" % (lambda_1_samples.mean(), lambda_1_samples.var()), color="#A60628", density=True)
plt.xlabel("Time (generations, n = 5 days)")
plt.ylabel("Density")
plt.title("Posterior Lambda 1 for %s" % county)
plt.axvline(lambda_1_samples.mean(), color='k', linestyle='dashed', linewidth=1)
plt.legend(loc = "upper right")


plt.subplot(2, 2, 3)
plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="mean = %.4f, var = %.4f" % (lambda_2_samples.mean(), lambda_2_samples.var()), color="#7A68A6", density=True)
plt.xlabel("Time (generations, n = 5 days)")
plt.ylabel("Density")
plt.title("Posterior Lambda 2 for %s" % county)
plt.axvline(lambda_2_samples.mean(), color='k', linestyle='dashed', linewidth=1)
plt.legend(loc = "upper right")

plt.subplot(2,2, 4)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(tau_samples, bins=len(R0_hat), alpha=1,
         label="mean = %.4f, var = %.4f" % (tau_samples.mean(), tau_samples.var()),
         color="#467821", weights=w, rwidth=2.)
plt.xlabel("Time (generations, n = 5 days)")
plt.ylabel("Density")
plt.title("Posterior Tau for %s" % county)
plt.axvline(tau_samples.mean(), color='k', linestyle='dashed', linewidth=1)
plt.legend(loc = "upper right")
plt.show()

## Summary Statistics:
print("Lambda 1: mean = %.4f, var = %.4f" % (np.mean(lambda_1_samples), np.var(lambda_1_samples)))
print("Lambda 2: mean = %.4f, var = %.4f" % (np.mean(lambda_2_samples), np.var(lambda_2_samples)))
print("Tau: mean = %.4f, var = %.4f " % (np.mean(tau_samples), np.var(tau_samples)))