# Gaussian Naive Baye's Classifier (Using Sci-Kit Learn Python Package)
import numpy as np
import pandas as pd
from math import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance

import os
dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)
## COVID-19 Severity Prediction
df = pd.read_csv('Mortality_incidence_sociodemographic_and_clinical_data_in_COVID19_patients.csv')

df_age = pd.get_dummies(df['Age'])*1
df = df.drop('Age', axis=1)
df = df.drop("CrtnScore", axis=1)
df = pd.concat([df, df_age], ignore_index=False, axis = 1)

resp = pd.DataFrame({"Severity":[1  if sev > 5 else  0 for sev in list(df['Severity'])]})
df = df.drop('Severity', axis=1)

## Model

X_train, X_test, y_train, y_test = train_test_split(df, resp, test_size = 0.3)

model = GaussianNB()

model.fit(X_train,y_train)
s = model.score(X_test,y_test)
print("Accuracy:", s)
r = permutation_importance(model, X_test, y_test, n_repeats = 20)

for i in r.importances_mean.argsort()[::-1]:
         print(f"{df.columns[i]:<8}"
               f"{r.importances_mean[i]:.3f}"
              f" +/-    {r.importances_std[i]:.3f}")
         

## Heart Disease Prediction
df = pd.read_csv('heart.csv')

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

## Model
X_train, X_test, y_train, y_test = train_test_split(df,resp,test_size=0.3)
model = GaussianNB()
model.fit(X_train,y_train)
s = model.score(X_test,y_test)
print("Accuracy:",s)

r = permutation_importance(model, X_test, y_test, n_repeats=30)
for i in r.importances_mean.argsort()[::-1]:
         print(f"{df.columns[i]:<8}"
               f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")