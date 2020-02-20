import pandas as pd
import numpy as np
from math import log

games = pd.read_csv("Data/Original/Games.csv")
#delete unnecessary columns
del games["Unnamed: 0"]
del games["game_id"]
#making wl have element True or False
for index, row in games.iterrows():
	if row["wl_home"] == "W":
		games.at[index, "wl_home"] = True
	else:
		games.at[index, "wl_home"] = False

#sorting by date
games["game_date_home"] = pd.to_datetime(games.game_date_home)
games = games.sort_values(by = "game_date_home")
#creating column with home team
games["Home_team"] = [matchup[:3] for matchup in games["matchup_home"]]
games["Away_team"] = [matchup[-3:] for matchup in games["matchup_home"]]
def get_team_data(games = games):
	#creating database with teams
	team_names = list(games.Home_team.unique())
	team_data = {}
	for team in team_names:
		team_data[team] = {"W/L": [],
						   "P Scored": [],
						   "P Against": []}

	return team_data

#defining variables for feature engineering
RUN_AVG_LEN = 15
#defining function of pythagorean w/l
def log_decay(length):
	decay = []
	for i in range(length):
		decay.append(log(length - i + 1))
	return 1 / np.array(decay)

def_decay = log_decay

def stand_exp(points_for, points_against, decay = 1.):
	return 13.9

def pythagenpat(points_for, points_against, decay = np.full(3, 1.)):
	return ((np.sum(points_for) + np.sum(points_against)) / np.sum(decay)) ** 0.285

def py_wl(points_for = [], points_against = [], wl_history = [],
		  decay = True, decay_function = def_decay,
		  exp_func = stand_exp, fill = 0.5):
	if not decay:
		decay_function = norm_decay

	if len(points_for) > 0 or len(points_against) > 0:
		decay = decay_function(len(points_against))
		exp = exp_func(points_for, points_against, decay = decay)
		py_points_for = np.sum(np.array(decay * points_for)) ** exp
		py_points_against = np.sum(np.array(decay * points_against)) ** exp
		return py_points_for / (py_points_for + py_points_against)
	return fill

def wl(points_for = [], points_against = [], wl_history = [],
	   decay = True, decay_function = def_decay, fill = 0.5):
	if not decay:
		decay_function = norm_decay

	wl_history = np.array(wl_history).astype("float")
	if wl_history.shape[0] > 0:
		decay = decay_function(len(wl_history))
		dec_wl_history = decay * wl_history
		return np.sum(dec_wl_history) / np.sum(decay)
	return fill

def update_db(functions = [(wl, "wl_d", {"decay": True}), (py_wl, "py_wl_d", {"decay": True}),
						   (wl, "wl", {"decay": False}), (py_wl, "py_wl", {"decay": False}),
						   (py_wl, "py_wl_d_pat", {"decay": True, "exp_func": pythagenpat}), (py_wl, "py_wl_pat", {"decay": False, "exp_func": pythagenpat})], dataframe = games):
	team_data = get_team_data(games = games)
	for index, row in dataframe.iterrows():
		h_points_scored, v_points_scored = (team_data[row["Home_team"]]["P Scored"],
											team_data[row["Away_team"]]["P Scored"])
		h_points_against, v_points_against = (team_data[row["Home_team"]]["P Against"],
											  team_data[row["Away_team"]]["P Against"])
		h_wl, v_wl = (team_data[row["Home_team"]]["W/L"],
					  team_data[row["Away_team"]]["W/L"])
		for function in functions:
			var = {"points_against": h_points_against, "points_for": h_points_scored, "wl_history": h_wl}
			function_var = {**var, **function[2]}
			dataframe.at[index, function[1] + "_home"] = function[0](**function_var)
			#doing it for visitor team
			var = {"points_against": v_points_against, "points_for": v_points_scored, "wl_history": v_wl}
			function_var = {**var, **function[2]}
			dataframe.at[index, function[1] + "_visitor"] = function[0](**function_var)

		#updating database
		team_data[row["Home_team"]]["W/L"].append(row["wl_home"])
		team_data[row["Away_team"]]["W/L"].append(not row["wl_home"])
		team_data[row["Home_team"]]["P Scored"].append(row["pts_home"])
		team_data[row["Away_team"]]["P Scored"].append(row["pts_away"])
		team_data[row["Home_team"]]["P Against"].append(row["pts_away"])
		team_data[row["Away_team"]]["P Against"].append(row["pts_home"])

	return dataframe

def main():
	data = update_db()
	data.to_csv(r"Data/NewFeatures/all_exp_var_all_decay_var_pat_285.csv")

if __name__ == "__main__":
	main()