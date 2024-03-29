import numpy as np
import pandas as pd

def load_auto(normalize=True):
	"""
	Load and preprocess the Auto dataset.

	Arguments:
	normalize -- whether to normalize the data

	Returns:
	X_hp_train -- training data for the 'Horsepower Only' model
	X_all_train -- training data for the 'All Features' model
	Y_train -- training labels
	Auto -- the original dataset
	"""

	# import data
	Auto = pd.read_csv('Auto.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()
	features = ['horsepower', 'cylinders', 'displacement', 'weight', 'acceleration', 'year', 'origin']

	# Normalize the data
	if normalize:
		Auto[features] = (Auto[features] - Auto[features].mean()) / Auto[features].std()


	# Extract relevant data features and split data for two models
	# Model 1: Horsepower Only
	# Model 2: All Features
	X_hp_train = Auto[['horsepower']].values.T
	X_all_train = Auto[features].values.T
	Y_train = Auto[['mpg']].values.reshape(1, -1)

	return X_hp_train, X_all_train, Y_train, Auto