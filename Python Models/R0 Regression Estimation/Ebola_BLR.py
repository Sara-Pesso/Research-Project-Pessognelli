import os
dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)

from math import *
import pymc as pm
from matplotlib import *
from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
from math import *
import pymc as pm
import arviz as az

import os
dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)

## Import COVID-19 Data (USA-- by county, beginning 1/22/2020)
import pandas as pd
county = "Ebola"
df = pd.read_csv(r"E:\Research Project\Python Models\Ebola_2016_WestAfrica\case_counts_ebola2016.csv")

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

### Regression
LnXn = list(np.log(Xn))
day = list(np.linspace(1, len(LnXn), num = len(LnXn), endpoint = True))

i = 0
while i < len(LnXn):
    if LnXn[i] == -inf:
        del LnXn[i]
        del day[i]
    else:
        i+=1  

X = np.column_stack((np.ones(len(day)), day))

n = X.shape[0]
k = X.shape[1]

Q,R = np.linalg.qr(X, mode= 'reduced')
R_inv = np.linalg.inv(R)
R_t_inv = np.linalg.inv(R.T)
V_b = np.matmul(R_inv, R_t_inv)
b_hat = np.matmul(R_inv, Q.T)
b_hat = b_hat.dot(LnXn)
df =n-k
residuals = X.dot(b_hat) - LnXn
s2 = (sum(residuals**2))/df
nsims = 10000
sigma = np.sqrt(1/np.random.gamma(df/2, 1/(df*s2/2), nsims))
beta = np.random.multivariate_normal([0 for _ in range(k)], V_b, nsims)

for i in range(len(beta)):
    m = beta[i]
    sig = sigma[i]
    beta[i] = sig*m + b_hat

plt.subplot(1,3,1)
plt.xlabel("Sigma (Standard Deviation)")
plt.ylabel("Frequency")
plt.title("Posterior for Sigma")
plt.hist(sigma, bins = 100, label="mean = %.4f, var = %.4f" % (np.mean(sigma), np.var(sigma)))
plt.axvline(np.mean(sigma), color='k', linestyle='dashed', linewidth=1)
plt.legend(loc = "upper right")


#intercept
plt.subplot(1,3,3)
plt.xlabel("alpha (intercept)")
plt.ylabel("Frequency")
plt.title("Posterior for alpha (intercept)")
plt.hist(beta[:,0], bins = 100, label="mean = %.4f, var = %.4f" % (np.mean(beta[:,0]), np.var(beta[:,0])))
plt.axvline(np.mean(beta[:,0]), color='k', linestyle='dashed', linewidth=1)
plt.legend(loc = "upper right")

#slope
plt.subplot(1,3,2)
plt.xlabel("beta (slope, m)")
plt.ylabel("Frequency")
plt.title("Posterior for beta (slope, m)")
plt.hist(beta[:,1], bins = 100, label="mean = %.4f, var = %.4f" % (np.mean(beta[:,1]), np.var(beta[:,1])))
plt.axvline(np.mean(beta[:,1]), color='k', linestyle='dashed', linewidth=1)
plt.legend(loc = "upper right")

plt.suptitle("%s" % county)
plt.show()

## beta[0] = alpha (intercept)
## beta[1] = beta (slope)
print(np.mean(sigma), np.mean(beta[:,0]), np.mean(beta[:,1]))
print("R0 derived from mean of beta0:",exp(np.mean(beta[:,1])))
#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################
X = np.array(day)
Y = np.array(LnXn)

ls_coef_ = np.cov(X, Y)[0,1]/np.var(X)
ls_intercept = Y.mean() - ls_coef_*X.mean()

plt.scatter(X, Y, c="k")
plt.xlabel("trading signal")
plt.ylabel("returns")
plt.title("Empirical returns vs trading signal")
plt.plot(X, ls_coef_*X + ls_intercept, label = "Least-squares line")
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())
plt.legend(loc="upper left")
# plt.show()
plt.close()

#We perform a simple Bayesian linear regression on this dataset. We look for a model like:
# R = alpha + beta*x+epsilon
# where alpha and beta are our unknown parameters and epsilson ~ N(0,sigma). The most common priors on 
#  alpha and beta are Normal priors. We will also assign a prior on sigma, so that sigma  
#  is uniform over 0 to 100.
print("X:",X)
#####################################
with pm.Model() as model:
    std = pm.Uniform("std", 0, 100)
    beta_ = pm.Normal("beta_", mu=0, sigma=100)
    alpha = pm.Normal("alpha", mu=0, sigma=100)
    mean = pm.Deterministic("mean", alpha + beta_*X)
    obs = pm.Normal("obs", mu=mean, sigma=std, observed=Y)
    trace = pm.sample(cores = 1, return_inferencedata=True)

az.plot_posterior(trace, var_names=["std", "beta_", "alpha"],figsize=(20, 4))
plt.show()

#############################################################################################################################################################################
#############################################################################################################################################################################
#############################################################################################################################################################################
## BLP vs. Bayesian Line
## Best Linear Predictor (Theorem 2.5.2 in GIP)
## L(X) = alpha + beta*X
x_bar = np.mean(day)
y_bar = np.mean(LnXn)
var_x = np.var(day)
var_y = np.var(LnXn)
cov_xy = np.cov(day, LnXn)[0][1]

alpha = y_bar - (cov_xy/var_x)*x_bar
beta_ = cov_xy/var_x
print("BLP results:", alpha, beta_)
## and thus our BLP is
def BLP(x):
    return y_bar + (cov_xy/var_x)*(x - x_bar)

## Print the estimated R0
slope_m = cov_xy/var_x
R0_hat = exp(slope_m)

print("Estimated R0:", R0_hat)

# Plot the BLP line !
from sklearn.metrics import r2_score
r2_blp = r2_score(LnXn, BLP(day))

plt.scatter(day, LnXn, label = "Raw Data")
print("N:", len(LnXn))
plt.plot(day, BLP(day), label = "BLP: R^2 = %.4f , R0 = %.4f" % (r2_blp ,R0_hat))
plt.xlabel("Generation, n")
plt.ylabel("ln(Xn)")
plt.title("Linearized Regression: %s" % county)

## PLot the Bayesian regression line using the means
def BLR(X):
    Y_hat = [np.mean(beta[:,1])*x + np.mean(beta[:,0]) for x in X] 
    return Y_hat

r2_blr = r2_score(LnXn, BLR(day))

plt.plot(day, BLR(day), label = "BLR: R^2 = %.4f , R0 = %.4f" % (r2_blr ,exp(np.mean(beta[:,1]))))

plt.legend(loc = "lower right")
plt.show()