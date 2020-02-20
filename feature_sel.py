import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = pd.read_csv(r"Data/NewFeatures/all_exp_var_all_decay_var.csv")

def mean_thresh(scores, alpha = 1):
	return np.mean(np.array(scores)) * alpha

#if amount_ft = None then it will only use the threshold
def select_k_best(data, score_func = chi2, amount_ft = None, threshold_func= mean_thresh,
				select_col = ["wl_d_home", "py_wl_d_home"], thresh_var = {"alpha": 1}):
	select_dat = data[select_col + ["wl_home"]]
	if amount_ft == None:
		amount_ft = len(select_col)

	bestfeatures = SelectKBest(score_func = score_func, k = amount_ft)
	fit  = bestfeatures.fit(select_dat[select_col].astype("float"),
							select_dat["wl_home"].astype("bool"))
	scores = fit.scores_
	thresh = threshold_func(scores, **thresh_var)
	scores_nan = np.where(scores > thresh, scores, np.nan)
	best_cols = np.asarray(select_col)[~np.isnan(scores_nan)]
	return best_cols

def main():
	#doing correlation heatmap
	heatmap_col = ["wl_home", "wl_d_home", "py_wl_d_home", "py_wl_home",
				   "py_wl_pat_home", "py_wl_d_pat_home"]
	basic_cor = data[heatmap_col].corr()
	plt.style.use("ggplot")
	plt.show(sns.heatmap(basic_cor, annot = True))
	#selectk best feature selection
	select_col = ["wl_d_home", "py_wl_d_home", "py_wl_home",
				  "py_wl_pat_home", "py_wl_d_pat_home",
				  "wl_visitor", "wl_d_visitor", "py_wl_d_visitor", "py_wl_visitor",
				  "py_wl_pat_visitor", "py_wl_d_pat_visitor"]
	select_dat = data[select_col + ["wl_home"]]
	bestfeatures = SelectKBest(score_func = chi2, k = len(select_col))
	fit  = bestfeatures.fit(select_dat[select_col].astype("float"), select_dat["wl_home"].astype("bool"))
	plt.style.use("ggplot")
	plt.bar(select_col, fit.scores_)
	plt.xlabel("Feature Names")
	plt.ylabel("Score")
	plt.title("Score of Features")
	plt.show()

if __name__ == "__main__":
	main()
	print(select_k_best(data, alpha = 1., select_col = ["wl_d_home", "py_wl_d_home", "py_wl_home",
								  						"py_wl_pat_home", "py_wl_d_pat_home",
								                        "wl_visitor", "wl_d_visitor", "py_wl_d_visitor", "py_wl_visitor",
								                        "py_wl_pat_visitor", "py_wl_d_pat_visitor"])) 