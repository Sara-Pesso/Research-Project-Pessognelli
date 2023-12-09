import pandas as pd
from math import *
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import seaborn as sns
import pytensor.tensor as pt
np.random.seed(42)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

### Heart Disease Risk Factors
df = pd.read_csv(r'E:\Research Project\Python Models\Gaussian Naive Bayes\heart.csv')

df_sex = pd.get_dummies(df['Sex'])*1
df = df.drop('Sex', axis = 1)
df_chestpain = pd.get_dummies(df['ChestPainType'])*1
df = df.drop('ChestPainType', axis = 1)
df_ecg = pd.get_dummies(df['RestingECG'])*1
df = df.drop('RestingECG', axis = 1)
df_st = pd.get_dummies(df['ST_Slope'])*1
df = df.drop('ST_Slope', axis = 1)

df_angina = pd.DataFrame({'ExerciseAngina':[1  if ang == "Y" else  0 for ang in list(df['ExerciseAngina'])]})
df = df.drop('ExerciseAngina', axis = 1)

resp = df['HeartDisease']
df = df.drop('HeartDisease', axis = 1)

df = pd.concat([df_sex, df_chestpain, df_ecg, df_st, df_angina, df], ignore_index=False, axis = 1)

X = np.array(df)
y = np.array(resp)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

Xt = pytensor.shared(X_train)
yt = pytensor.shared(y_train)
n_classes = 2
n_features = X.shape[1]


# with pm.Model() as model:
#     β = pm.Normal("β", 0, sigma=1e2, shape=(n_features, n_classes))
#     a = pm.Normal("a", sigma=1e4, shape=(n_classes,))
#     p = pt.special.softmax(Xt.dot(β) + a, axis=-1)

#     observed = pm.Categorical("obs", p=p, observed=yt)

#     sampler = pm.NUTS()
#     trace_log = pm.sample(draws=1000, step = sampler, cores=1, chains=2, tune=1000, random_seed=100)

# az.plot_posterior(trace_log, var_names=["β", "a"],figsize=(40, 4))
# plt.show()

# stat_df = az.summary(trace_log)

# def LogRegPredictions(x):
#     pxs = []
#     for n in range(0,n_classes):
#         #beta[beta i, class j]
#         px = 0
#         for i in range(0, n_features):
#             varname = "β[%s, %s]" % (i, n)
#             b_hat = stat_df["mean"][varname]
#             px += b_hat*x[i]
        
#         varname = "a[%s]" % n
#         px += stat_df["mean"][varname]
#         pxs.append(px)

#     class_prediction = np.argmax(pxs)  
#     return class_prediction

# predictions = []
# for j in range(0, len(X_test)):
#     m = LogRegPredictions(X_test[j])
#     # print(m, y[j])
#     predictions.append(m)

# accuracy = sum(predictions == y_test)/len(y_test)
# print("Accuracy:", accuracy)
# print("Summary of Posterior Distributions for all a & β:")
# print(stat_df)

### COVID-19 Risk Factors
df = pd.read_csv(r'E:\Research Project\Python Models\Gaussian Naive Bayes\Mortality_incidence_sociodemographic_and_clinical_data_in_COVID19_patients.csv')

df_age = pd.get_dummies(df['Age'])*1
df = df.drop('Age', axis=1)
df = pd.concat([df, df_age], ignore_index=False, axis = 1)

resp = pd.DataFrame({"Severity":[1  if sev > 3 else  0 for sev in list(df['Severity'])]})
df = df.drop('Severity', axis=1)

## Model
X = np.array(df)
y = np.array(resp["Severity"])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

Xt = pytensor.shared(X_train)
yt = pytensor.shared(y_train)

n_classes = 2
n_features = X.shape[1]

with pm.Model() as model:
    β = pm.Normal("β", 0, sigma=1e4, shape=(n_features, n_classes))
    a = pm.Normal("a", sigma=1e6, shape=(n_classes,))
    p = pt.special.softmax(Xt.dot(β) + a, axis=-1)
    observed = pm.Categorical("obs", p=p, observed=yt)
    sampler = pm.NUTS()
    trace_log = pm.sample(draws=1000, step = sampler, cores=1, chains=2, tune=1000, random_seed=100)

az.plot_posterior(trace_log, var_names=["β", "a"],figsize=(40, 4))
plt.show()

stat_df = az.summary(trace_log)

def LogRegPredictions(x):
    pxs = []
    for n in range(0,n_classes):
        #beta[beta i, class j]
        px = 0
        for i in range(0, n_features):
            varname = "β[%s, %s]" % (i, n)
            b_hat = stat_df["mean"][varname]
            px += b_hat*x[i]
        
        varname = "a[%s]" % n
        px += stat_df["mean"][varname]
        pxs.append(px)

    class_prediction = np.argmax(pxs)  
    return class_prediction

predictions = []
for j in range(0, len(X_test)):
    m = LogRegPredictions(X_test[j])
    # print(m, y[j])
    predictions.append(m)

accuracy = sum(predictions == y_test)/len(y_test)
print("Accuracy:", accuracy)
print("Summary of Posterior Distributions for all a & β:")
print(stat_df)

