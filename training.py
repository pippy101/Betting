from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
import feature_sel
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

data = pd.read_csv(r"Data/NewFeatures/all_exp_var_all_decay_var.csv")

def normailize_data_model(data, norm_col = ["wl_d_home", "py_wl_d_home"], target_col = ["wl_home"],
					      model = LogisticRegression, model_par = {"C": 10.}):
	models = [model(**model_par) for i in range(len(norm_col))]
	norm_data = pd.DataFrame()
	for m in range(len(models)): 
		models[m].fit(np.asarray(data[norm_col[m]]).reshape(-1, 1), data[target_col].astype("int")) 
		norm_data[norm_col[m]] = model[m].predict(np.asarray(data[norm_col[m]]).reshape(-1, 1))
	print(norm_data)

normailize_data_model(data)