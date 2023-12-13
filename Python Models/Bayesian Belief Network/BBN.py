import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator

# Gaussian Naive Baye's Classifier (Using Sci-Kit Learn Python Package)
import numpy as np
import pandas as pd
from math import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
from pgmpy.estimators import HillClimbSearch, ExhaustiveSearch, BicScore, BDeuScore, K2Score
from pgmpy.metrics import structure_score


import os
dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)


df = pd.read_csv('heart.csv')

resp = df['HeartDisease']
df_abnormalECG = pd.DataFrame({'Abnormal ECG':[1  if ecg != "Normal" else  0 for ecg in list(df['RestingECG'])]})
df_ChestPain = pd.DataFrame({'Chest Pain':[1  if ecg != "ASY" else  0 for ecg in list(df['ChestPainType'])]})
df_cholesterol = pd.DataFrame({'High Cholesterol':[1  if float(ch) >= 200 else  0 for ch in list(df['Cholesterol'])]})
df_bloodpressure = pd.DataFrame({'High Blood Pressure':[1  if float(bp) >= 130 else  0 for bp in list(df['RestingBP'])]})

data = pd.concat([df_ChestPain, df_abnormalECG, df_cholesterol, df_bloodpressure, resp], ignore_index=False, axis = 1)
# data = pd.concat([df_cholesterol, df_bloodpressure, resp], ignore_index=False, axis = 1)

scorer = 'bic'
model = BayesianModel([('Chest Pain', 'Abnormal ECG'),('High Cholesterol', 'Chest Pain'),('High Cholesterol', 'HeartDisease'), ('High Cholesterol', 'High Blood Pressure'), ('High Blood Pressure', 'HeartDisease')])
print("Proposed DAG score:", structure_score(model, data, scoring_method=scorer))
# model = BayesianModel([('High Cholesterol', 'High Blood Pressure'), ('High Blood Pressure', 'HeartDisease'), ('High Cholesterol', 'HeartDisease')])

# model.fit(data, estimator=BayesianEstimator, prior_type="BDeu") # default equivalent_sample_size=5
# for cpd in model.get_cpds():
#     print(cpd)

# print("High Blood Pressure(0) = ", sum(df_bloodpressure["High Blood Pressure"] == 0)/len(list(df_bloodpressure["High Blood Pressure"])))
# print("High Blood Pressure(1) = ", sum(df_bloodpressure["High Blood Pressure"] == 1)/len(list(df_bloodpressure["High Blood Pressure"])))

bic = BicScore(data)
es = ExhaustiveSearch(data, scoring_method=bic)
best_model = es.estimate()

print("Exhaustive Search (BIC):",best_model.edges())
print("Exhaustive Search BIC Score:", structure_score(BayesianModel(best_model.edges()), data, scoring_method=scorer))

hc = HillClimbSearch(data)
best_model = hc.estimate(scoring_method=bic)
print("Hill Climb Search (BIC):",best_model.edges())
print("Hill Climb Search BIC Score:", structure_score(BayesianModel(best_model.edges()), data, scoring_method=scorer))

