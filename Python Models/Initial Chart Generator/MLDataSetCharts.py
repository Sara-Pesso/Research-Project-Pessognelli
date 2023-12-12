import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import os
dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)
## COVID-19 Severity Prediction
df = pd.read_csv('Mortality_incidence_sociodemographic_and_clinical_data_in_COVID19_patients.csv')

fig = px.scatter_matrix(df)
fig.show()
## Heart Disease Prediction
df = pd.read_csv('heart.csv')

fig = px.scatter_matrix(df)
fig.show()